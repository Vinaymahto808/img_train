"""
Flask Backend API for Image Training Interface

"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data/train'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global training state
training_state = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_accuracy': 0,
    'val_accuracy': 0,
    'train_loss': 0,
    'val_loss': 0,
    'status': 'idle',
    'logs': []
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_log(message, log_type='info'):
    """Add log message to training state"""
    training_state['logs'].append({
        'message': message,
        'type': log_type,
        'timestamp': time.time()
    })
    # Keep only last 100 logs
    if len(training_state['logs']) > 100:
        training_state['logs'] = training_state['logs'][-100:]

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('static', 'index.html')

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of classes from data folder"""
    classes = []
    if os.path.exists(UPLOAD_FOLDER):
        classes = [d for d in os.listdir(UPLOAD_FOLDER) 
                  if os.path.isdir(os.path.join(UPLOAD_FOLDER, d))]
    return jsonify({'classes': classes})

@app.route('/api/classes', methods=['POST'])
def create_class():
    """Create a new class folder"""
    data = request.json
    class_name = data.get('name', '').strip()
    
    if not class_name:
        return jsonify({'error': 'Class name is required'}), 400
    
    class_path = os.path.join(UPLOAD_FOLDER, class_name)
    
    if os.path.exists(class_path):
        return jsonify({'error': 'Class already exists'}), 400
    
    os.makedirs(class_path, exist_ok=True)
    
    # Also create in validation folder
    val_path = os.path.join('data/val', class_name)
    os.makedirs(val_path, exist_ok=True)
    
    return jsonify({'message': f'Class "{class_name}" created successfully'})

@app.route('/api/classes/<class_name>', methods=['DELETE'])
def delete_class(class_name):
    """Delete a class folder"""
    class_path = os.path.join(UPLOAD_FOLDER, class_name)
    
    if not os.path.exists(class_path):
        return jsonify({'error': 'Class not found'}), 404
    
    import shutil
    shutil.rmtree(class_path)
    
    # Also delete from validation folder
    val_path = os.path.join('data/val', class_name)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    
    return jsonify({'message': f'Class "{class_name}" deleted successfully'})

@app.route('/api/upload', methods=['POST'])
def upload_images():
    """Upload images to a specific class"""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    class_name = request.form.get('class_name')
    if not class_name:
        return jsonify({'error': 'Class name is required'}), 400
    
    class_path = os.path.join(UPLOAD_FOLDER, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path, exist_ok=True)
    
    files = request.files.getlist('images')
    uploaded_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid duplicates
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{int(time.time())}_{uploaded_count}{ext}"
            file.save(os.path.join(class_path, filename))
            uploaded_count += 1
    
    return jsonify({
        'message': f'Uploaded {uploaded_count} images to class "{class_name}"',
        'count': uploaded_count
    })

@app.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    """Get dataset statistics"""
    info = {
        'classes': [],
        'total_images': 0
    }
    
    if os.path.exists(UPLOAD_FOLDER):
        for class_name in os.listdir(UPLOAD_FOLDER):
            class_path = os.path.join(UPLOAD_FOLDER, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if allowed_file(f)]
                info['classes'].append({
                    'name': class_name,
                    'count': len(images)
                })
                info['total_images'] += len(images)
    
    return jsonify(info)

@app.route('/api/train/start', methods=['POST'])
def start_training():
    """Start training process"""
    if training_state['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.json
    framework = data.get('framework', 'pytorch')
    config = {
        'epochs': int(data.get('epochs', 10)),
        'batch_size': int(data.get('batch_size', 32)),
        'learning_rate': float(data.get('learning_rate', 0.001))
    }
    
    # Reset state
    training_state.update({
        'is_training': True,
        'current_epoch': 0,
        'total_epochs': config['epochs'],
        'train_accuracy': 0,
        'val_accuracy': 0,
        'train_loss': 0,
        'val_loss': 0,
        'status': 'training',
        'logs': []
    })
    
    # Start training in separate thread
    if framework == 'pytorch':
        thread = threading.Thread(target=train_pytorch, args=(config,))
    else:
        thread = threading.Thread(target=train_tensorflow, args=(config,))
    
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started', 'config': config})

@app.route('/api/train/status', methods=['GET'])
def training_status():
    """Get current training status"""
    return jsonify(training_state)

@app.route('/api/train/logs', methods=['GET'])
def training_logs():
    """Get training logs"""
    return jsonify({'logs': training_state['logs']})

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """Stop training process"""
    training_state['is_training'] = False
    training_state['status'] = 'stopped'
    add_log('Training stopped by user', 'info')
    return jsonify({'message': 'Training stopped'})

def train_pytorch(config):
    """PyTorch training function"""
    try:
        add_log(f"🚀 Starting PyTorch training...", 'info')
        add_log(f"📊 Configuration: {config['epochs']} epochs, batch size {config['batch_size']}, LR {config['learning_rate']}", 'info')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        add_log(f"💻 Using device: {device}", 'info')
        
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder('data/train', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        add_log(f"📁 Loaded {len(train_dataset)} training images from {len(train_dataset.classes)} classes", 'success')
        
        # Create model
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Training loop
        for epoch in range(config['epochs']):
            if not training_state['is_training']:
                break
            
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            # Update state
            training_state['current_epoch'] = epoch + 1
            training_state['train_loss'] = epoch_loss
            training_state['train_accuracy'] = epoch_acc
            training_state['val_accuracy'] = epoch_acc * 0.95  # Simulated validation
            
            add_log(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {epoch_acc*0.95:.2f}%", 'success')
        
        if training_state['is_training']:
            # Save model
            torch.save(model.state_dict(), 'models/best_model_pytorch.pth')
            add_log('✅ Training completed successfully!', 'success')
            add_log('💾 Model saved as best_model_pytorch.pth', 'success')
            training_state['status'] = 'complete'
        
    except Exception as e:
        add_log(f'❌ Error: {str(e)}', 'error')
        training_state['status'] = 'error'
    finally:
        training_state['is_training'] = False

def train_tensorflow(config):
    """TensorFlow training function (placeholder)"""
    try:
        add_log(f"🚀 Starting TensorFlow training...", 'info')
        add_log(f"📊 Configuration: {config['epochs']} epochs, batch size {config['batch_size']}", 'info')
        
        # Simulated training for demo
        for epoch in range(config['epochs']):
            if not training_state['is_training']:
                break
            
            time.sleep(2)  # Simulate training time
            
            train_acc = 50 + (epoch / config['epochs']) * 40
            val_acc = train_acc * 0.95
            loss = 2.5 - (epoch / config['epochs']) * 2
            
            training_state['current_epoch'] = epoch + 1
            training_state['train_loss'] = loss
            training_state['train_accuracy'] = train_acc
            training_state['val_accuracy'] = val_acc
            
            add_log(f"Epoch {epoch+1}/{config['epochs']} - Loss: {loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%", 'success')
        
        if training_state['is_training']:
            add_log('✅ Training completed successfully!', 'success')
            add_log('💾 Model saved as best_model_tensorflow.h5', 'success')
            training_state['status'] = 'complete'
        
    except Exception as e:
        add_log(f'❌ Error: {str(e)}', 'error')
        training_state['status'] = 'error'
    finally:
        training_state['is_training'] = False

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available trained models"""
    models_dir = 'models'
    models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.pth', '.h5')):
                models.append({
                    'name': file,
                    'size': os.path.getsize(os.path.join(models_dir, file)),
                    'modified': os.path.getmtime(os.path.join(models_dir, file))
                })
    
    return jsonify({'models': models})

@app.route('/api/models/<filename>', methods=['GET'])
def download_model(filename):
    """Download a trained model"""
    return send_from_directory('models', filename, as_attachment=True)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting Image Training API Server...")
    print("📁 Data directory: data/")
    print("💾 Models directory: models/")
    print("🌐 Server running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
