import layers
from model import MyAI
import numpy as np
import data_loader as dl

def predict(model, x):
    return model.forward(x)

def evaluate(model, data, labels):
    correct = 0
    for i in range(len(data)):
        pred = predict(model, data[i])
        if np.argmax(pred) == np.argmax(labels[i]):
            correct += 1
    return correct / len(data)

def train(model, data, labels, epochs=10, lr=0.001, lr_decay=0.95):
    loss_fn = layers.CrossEntropyLossLayer()
    current_lr = lr

    for epoch in range(epochs):
        total_loss = 0
        
        #Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        ladels = labels[indices]

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

if __name__ == "__main__":
    dataset_path = r"D:\Programing_materials\Python\python_Projects\Image_Identify_CNN\Dataset"

    data, labels, class_names = dl.load_dataset(dataset_path, max_classes=3)

    print("Classes:", class_names)
    print("Data shape:", data.shape)
    
    model = MyAI(num_classes=len(class_names))

    train(model, data, labels, epochs=20, lr=0.001, lr_decay=0.95)