import layers
from model import MyAI
from optimizer import AdamW
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
    
    # Batch size
    parser.add_argument("--batch-size",   type=int,                        default=64,                         help="Batch size (default: 64)")

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

    # Dropout
    parser.add_argument("--dropout-prob", type=float,                      default=0.3,                        help="Dropout probability (default: 0.3)")
    return parser.parse_args()

def predict(model, x):
    return model.forward(x)

def split_data(data, labels, val_ratio=0.2):
    indices = np.random.permutation(len(data))
    split = int(len(data) * (1 - val_ratio))
    train_idx = indices[:split]
    val_idx   = indices[split:]
    return data[train_idx], labels[train_idx], data[val_idx], labels[val_idx]

def evaluate(model, data, labels, batch_size=64):
    correct = 0
    for i in range(0, len(data), batch_size):
        end = min(i + batch_size, len(data))

        batch_data = data[i:end]
        batch_labels = labels[i:end]

        prediction = model.forward(batch_data)

        pred_classes = np.argmax(prediction, axis=1)
        target_classes = np.argmax(batch_labels, axis=1)

        correct += np.sum(pred_classes == target_classes)

    return correct / len(data)

def save_model(model, path="model.npz"):
    np.savez(path,
        #Conv
        conv1_kernels=model.conv1.kernel_data,
        conv1_bias=model.conv1.bias_data,
        conv2_kernels=model.conv2.kernel_data,
        conv2_bias=model.conv2.bias_data,
        conv3_kernels=model.conv3.kernel_data,
        conv3_bias=model.conv3.bias_data,

        #FC
        fc1_weights=model.fc1.weights,
        fc1_bias=model.fc1.bias,
        fc2_weights=model.fc2.weights,   
        fc2_bias=model.fc2.bias,

        #BN
        bn1_gamma=model.bn1.gamma,
        bn1_beta=model.bn1.beta,
        bn1_running_mean=model.bn1.running_mean,
        bn1_running_var=model.bn1.running_var,

        bn2_gamma=model.bn2.gamma,
        bn2_beta=model.bn2.beta,
        bn2_running_mean=model.bn2.running_mean,
        bn2_running_var=model.bn2.running_var,

        bn3_gamma=model.bn3.gamma,
        bn3_beta=model.bn3.beta,
        bn3_running_mean=model.bn3.running_mean,
        bn3_running_var=model.bn3.running_var,
    )
    print(f"Model saved to {path}")

def load_model(model, path="model.npz"):
    d = np.load(path)
    model.conv1.kernel_data = d["conv1_kernels"]
    model.conv1.bias_data   = d["conv1_bias"]
    model.conv2.kernel_data = d["conv2_kernels"]
    model.conv2.bias_data   = d["conv2_bias"]
    model.conv3.kernel_data = d["conv3_kernels"]
    model.conv3.bias_data   = d["conv3_bias"]
    model.fc1.weights = d["fc1_weights"]
    model.fc1.bias    = d["fc1_bias"]
    model.fc2.weights = d["fc2_weights"]
    model.fc2.bias    = d["fc2_bias"]
    model.fc1.weights_gradient = np.zeros_like(model.fc1.weights)
    model.fc1.bias_gradient = np.zeros_like(model.fc1.bias)

    # model.fc1.weights_velocity = np.zeros_like(model.fc1.weights)
    # model.fc1.bias_velocity = np.zeros_like(model.fc1.bias)

    model.fc2.weights_gradient = np.zeros_like(model.fc2.weights)
    model.fc2.bias_gradient = np.zeros_like(model.fc2.bias)

    # model.fc2.weights_velocity = np.zeros_like(model.fc2.weights)
    # model.fc2.bias_velocity = np.zeros_like(model.fc2.bias)

    # BN1
    model.bn1.gamma = d["bn1_gamma"]
    model.bn1.beta = d["bn1_beta"]
    model.bn1.running_mean = d["bn1_running_mean"]
    model.bn1.running_var = d["bn1_running_var"]

    # BN2
    model.bn2.gamma = d["bn2_gamma"]
    model.bn2.beta = d["bn2_beta"]
    model.bn2.running_mean = d["bn2_running_mean"]
    model.bn2.running_var = d["bn2_running_var"]

    # BN3
    model.bn3.gamma = d["bn3_gamma"]
    model.bn3.beta = d["bn3_beta"]
    model.bn3.running_mean = d["bn3_running_mean"]
    model.bn3.running_var = d["bn3_running_var"]
    
    print(f"Model loaded from {path}")
    return model

def train(model, 
          train_data, 
          train_labels, 
          batch_size=64,
          epochs=10, 
          lr=0.001, 
          lr_decay=0.98,
          l2_lambda=0.0001,
          save=True, 
          save_to="model.npz", 
          val_data=None, 
          val_labels=None):
    
    loss_fn = layers.CrossEntropyLossLayer()

    # Complete lazy init before optimizer.
    model.forward(train_data[:1])
    optimizer = AdamW(model, learning_rate=lr, weight_decay=l2_lambda)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        #Shuffle
        indices = np.random.permutation(len(train_data))

        for start in range(0, len(train_data), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            input_data = train_data[batch_indices]
            target = train_labels[batch_indices]

            prediction = model.forward(input_data)
            loss = loss_fn.forward(prediction, target)
            total_loss += loss * len(batch_indices)

            grad = loss_fn.backward(prediction, target)

            model.backward(grad)

            optimizer.step()
        
        train_loss = total_loss / len(train_data)
        model.eval()
        train_acc = evaluate(model, train_data, train_labels, batch_size=batch_size)
        model.train()
        
        if val_data is not None:
            model.eval()
            val_loss = 0
            correct = 0

            for start in range(0, len(val_data), batch_size):
                end = min(start + batch_size, len(val_data))

                batch_data = val_data[start:end]
                batch_target = val_labels[start:end]

                pred = model.forward(batch_data)

                loss = loss_fn.forward(pred, batch_target)

                val_loss += loss * len(batch_data)

                pred_classes = np.argmax(pred, axis=1)
                target_classes = np.argmax(batch_target, axis=1)

                correct += np.sum(
                    pred_classes == target_classes
                )

            val_loss /= len(val_data)
            val_acc = correct / len(val_data)

            gap = train_acc - val_acc
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.4%}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_acc:.4%}, Gap: {gap:.6f}, LR: {optimizer.lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save:
                    save_model(model, save_to)
                    print(f"  → New best saved! Val: {val_acc:.2%}")
        else:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Accuracy: {train_acc:.4%}, LR: {optimizer.lr:.6f}")
        
        optimizer.lr *= lr_decay

if __name__ == "__main__":
    args = parse_args()

    data, labels, class_names = dl.load_dataset(args.dataset_path, 
                                                max_classes=args.num_classes)
    train_data, train_labels, val_data, val_labels = split_data(data, labels, val_ratio=0.2)
    
    print("Classes:", class_names)
    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")
    print("Data shape:", data.shape)
    
    model = MyAI(num_classes=len(class_names), dropout_prob=args.dropout_prob)

    if args.loading and not args.no_loading:
        model.forward(data[0:1])
        load_model(model, args.load_from)

    should_save = args.saving and not args.no_saving
    train(model, 
          train_data, 
          train_labels,
          batch_size=args.batch_size,
          epochs=args.epochs,
          lr=args.lr,
          lr_decay=args.lr_decay,
          l2_lambda=args.l2_lambda,
          save=should_save,
          save_to=args.save_to,
          val_data=val_data,
          val_labels=val_labels)