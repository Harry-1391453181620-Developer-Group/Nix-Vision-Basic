import layers
from model import MyAI
import numpy as np
import data_loader as dl

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train MyAI CNN")

    # Dataset
    parser.add_argument("--dataset-path", type=str, required=True,      help="Path to dataset directory")
    parser.add_argument("--num-classes",  type=int, default=None,       help="Number of classes to load (default: all)")

    parser.add_argument("--lr",        type=float, default=0.001,       help="Learning rate")
    parser.add_argument("--lr-decay",  type=float, default=0.98,        help="LR decay per epoch")
    parser.add_argument("--epochs",    type=int,   default=40,          help="Number of epochs")

    # Saving
    parser.add_argument("--saving",    action="store_true",             help="Enable model saving")
    parser.add_argument("--no-saving", action="store_true",             help="Disable model saving")
    parser.add_argument("--save-to",   type=str,   default="model.npz", help="Path to save model (only used with --saving)")

    # Loading
    parser.add_argument("--loading",   action="store_true",             help="Load model before training")
    parser.add_argument("--no-loading",action="store_true",             help="Explicitly start fresh (default)")
    parser.add_argument("--load-from", type=str,   default="model.npz", help="Path to load model from (only used with --loading)")

    return parser.parse_args()

def predict(model, x):
    return model.forward(x)

def evaluate(model, data, labels):
    correct = 0
    for i in range(len(data)):
        pred = predict(model, data[i])
        if np.argmax(pred) == np.argmax(labels[i]):
            correct += 1
    return correct / len(data)

def save_model(model, path="model.npz"):
    np.savez(path,
        # Conv layer
        conv_kernels=model.conv.kernel_data,
        conv_bias=model.conv.bias_data,
        # FC layer
        fc_weights=model.fc.weights,
        fc_bias=model.fc.bias
    )
    print(f"Model saved to {path}")

def load_model(model, path="model.npz"):
    data = np.load(path)
    model.conv.kernel_data = data["conv_kernels"]
    model.conv.bias_data = data["conv_bias"]
    model.fc.weights = data["fc_weights"]
    model.fc.bias = data["fc_bias"]
    print(f"Model loaded from {path}")
    return model

def train(model, data, labels, epochs=10, lr=0.001, lr_decay=0.98, save=True, save_to="model.npz"):
    loss_fn = layers.CrossEntropyLossLayer()
    current_lr = lr

    for epoch in range(epochs):
        total_loss = 0
        
        #Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]

        for i in range(len(data)):
            input_data = data[i]
            target = labels[i].reshape(1, -1)

            # Forward
            prediction = model.forward(input_data)

            # Loss
            loss = loss_fn.forward(prediction, target)
            total_loss += loss

            # Backward
            grad = loss_fn.backward(prediction, target)
            model.backward(grad)

            # Update
            model.update(current_lr)

        avg_loss = total_loss / len(data)
        accuracy = evaluate(model, data, labels)
        
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2%}, LR: {current_lr:.6f}")
        current_lr *= lr_decay

    if save:
        save_model(model, save_to)

if __name__ == "__main__":
    args = parse_args()

    data, labels, class_names = dl.load_dataset(args.dataset_path, max_classes=args.num_classes)

    print("Classes:", class_names)
    print("Data shape:", data.shape)
    
    model = MyAI(num_classes=len(class_names))

    if args.loading and not args.no_loading:
        model.forward(data[0])
        load_model(model, args.load_from)

    should_save = args.saving and not args.no_saving
    train(model, data, labels,
          epochs=args.epochs,
          lr=args.lr,
          lr_decay=args.lr_decay,
          save=should_save,
          save_to=args.save_to)