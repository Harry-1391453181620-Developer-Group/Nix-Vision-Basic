import numpy as np
from PIL import Image
from model import MyAI
from layers import *

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
        print(f"Model loaded from {path}")

def load_image(path):
    img = Image.open(path).convert("L")  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to 64x64
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = img_array[np.newaxis, :, :]  # Add batch and channel dimensions
    return img_array

def enable_winograd(model):
    model.conv1.algorithm = "winograd"
    model.conv2.algorithm = "winograd"
    model.conv3.algorithm = "winograd"

    model.conv1.wino_ready = False
    model.conv2.wino_ready = False
    model.conv3.wino_ready = False

def predict(model, image):
    model.dropout.eval()
    output = model.forward(image)
    return np.argmax(output)

if __name__ == "__main__":
    model = MyAI(num_classes=13)

    load_model(model, "model.npz")
    enable_winograd(model)

    img = load_image("image.png")
    pred_class = predict(model, img)

    print(f"Predicted class: {pred_class}")
            