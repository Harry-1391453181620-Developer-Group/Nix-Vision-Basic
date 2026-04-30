import layers
from model import MyAI
import numpy as np
import data_loader as dl
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train MyAI CNN")

    # Dataset
    parser.add_argument("--dataset-path", type=str,                                             required=True, help="Path to dataset directory")
    parser.add_argument("--num-classes",  type=int,                        default=None,                       help="Number of classes to load (default: all)")

    # Epoch number         
    parser.add_argument("--epochs",       type=int,                        default=40,                         help="Number of epochs")
            
    # Learning rate                  
    parser.add_argument("--lr",           type=float,                      default=0.001,                      help="Learning rate")
    parser.add_argument("--lr-decay",     type=float,                      default=0.98,                       help="LR decay per epoch")
    parser.add_argument("--l2-lambda",    type=float,                      default=0.0001,                     help="L2 regularization strength")
    
    # Saving                    
    parser.add_argument("--saving",                   action="store_true",                                     help="Enable model saving")
    parser.add_argument("--no-saving",                action="store_true",                                     help="Disable model saving")
    parser.add_argument("--save-to",      type=str,                        default="model.npz",                help="Path to save model (only used with --saving")
    # Loading           
    parser.add_argument("--loading",                  action="store_true",                                     help="Load model before training")
    parser.add_argument("--no-loading",               action="store_true",                                     help="Explicitly start fresh (default)")
    parser.add_argument("--load-from",    type=str,                        default="model.npz",                help="Path to load model from (only used with --loading)")

    return parser.parse_args()

def predict(model, x):
    return model.predict(x)

def split_data(data, labels, val_ratio=0.2):
    indices = np.random.permutation(len(data))
    split = int(len(data) * (1 - val_ratio))
    train_idx = indices[:split]
    val_idx   = indices[split:]
    return data[train_idx], labels[train_idx], data[val_idx], labels[val_idx]

def evaluate(model, data, labels):
    correct = 0
    for i in range(len(data)):
        pred = predict(model, data[i])
        if np.argmax(pred) == np.argmax(labels[i]):
            correct += 1
    return correct / len(data)

def save_model(model, path="model.npz"):
    np.savez(path,
        #Conv
        conv1_kernels=model.conv1.kernel_data,
        conv1_bias=model.conv1.bias_data,
        conv2_kernels=model.conv2.kernel_data,
        conv2_bias=model.conv2.bias_data,

        #FC
        fc1_weights=model.fc1.weights,
        fc1_bias=model.fc1.bias,
        fc2_weights=model.fc2.weights,   
        fc2_bias=model.fc2.bias
    )
    print(f"Model saved to {path}")

def load_model(model, path="model.npz"):
    d = np.load(path)
    model.conv1.kernel_data = d["conv1_kernels"]
    model.conv1.bias_data   = d["conv1_bias"]
    model.conv2.kernel_data = d["conv2_kernels"]
    model.conv2.bias_data   = d["conv2_bias"]
    model.fc1.weights = d["fc1_weights"]
    model.fc1.bias    = d["fc1_bias"]
    model.fc2.weights = d["fc2_weights"]
    model.fc2.bias    = d["fc2_bias"]
    print(f"Model loaded from {path}")
    return model

def train(model, 
          train_data, 
          train_labels, 
          epochs=10, 
          lr=0.001, 
          lr_decay=0.98,
          l2_lambda=0.0001,
          save=True, 
          save_to="model.npz", 
          val_data=None, 
          val_labels=None):
    
    loss_fn = layers.CrossEntropyLossLayer()
    current_lr = lr
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.dropout.train()
        total_loss = 0
        
        #Shuffle
        indices = np.random.permutation(len(train_data))
        for idx in indices:
            input_data = train_data[idx]
            target = train_labels[idx].reshape(1, -1)

            prediction = model.forward(input_data)
            loss = loss_fn.forward(prediction, target)
            total_loss += loss

            grad = loss_fn.backward(prediction, target)
            model.backward(grad)

            model.update(current_lr, l2_lambda=l2_lambda)
        
        train_loss = total_loss / len(train_data)
        train_acc = evaluate(model, train_data, train_labels)
        
        if val_data is not None:
            model.dropout.eval()
            val_loss = 0
            correct = 0

            for i in range(len(val_data)):
                pred = model.forward(val_data[i])
                val_loss += loss_fn.forward(pred, val_labels[i].reshape(1, -1))

                if np.argmax(pred) == np.argmax(val_labels[i]):
                    correct += 1

            val_loss /= len(val_data)
            val_acc = correct / len(val_data)

            gap = train_acc - val_acc
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.4%}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_acc:.4%}, Gap: {gap:.6f}, LR: {current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save:
                    save_model(model, save_to)
                    print(f"  → New best saved! Val: {val_acc:.2%}")
        else:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Accuracy: {train_acc:.4%}, LR: {current_lr:.6f}")
        
        current_lr *= lr_decay

if __name__ == "__main__":
    args = parse_args()

    data, labels, class_names = dl.load_dataset(args.dataset_path, 
                                                max_classes=args.num_classes)
    train_data, train_labels, val_data, val_labels = split_data(data, labels, val_ratio=0.2)
    
    print("Classes:", class_names)
    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")
    print("Data shape:", data.shape)
    
    model = MyAI(num_classes=len(class_names))

    if args.loading and not args.no_loading:
        model.forward(data[0])
        load_model(model, args.load_from)

    should_save = args.saving and not args.no_saving
    train(model, train_data, train_labels,
          epochs=args.epochs,
          lr=args.lr,
          lr_decay=args.lr_decay,
          l2_lambda=args.l2_lambda,
          save=should_save,
          save_to=args.save_to,
          val_data=val_data,
          val_labels=val_labels)