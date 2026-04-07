import layers
from model import MyAI
import numpy as np
import data_loader as dl

def train(model, data, labels, epochs=10, lr=0.01):
    loss_fn = layers.CrossEntropyLossLayer()

    for epoch in range(epochs):
        total_loss = 0

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
            model.update(lr)

        print(f"Epoch {epoch}, Loss: {total_loss / len(data)}")
        if epoch % 5 == 0 and i == 0:
            print("Prediction:", prediction)

if __name__ == "__main__":
    dataset_path = r"D:\Programing_materials\Python\python_Projects\Image_Identify_CNN\Dataset"

    data, labels, class_names = dl.load_dataset(dataset_path, max_classes=3)

    print("Classes:", class_names)
    print("Data shape:", data.shape)
    
    model = MyAI(num_classes=len(class_names))

    train(model, data, labels, epochs=20, lr=0.001)