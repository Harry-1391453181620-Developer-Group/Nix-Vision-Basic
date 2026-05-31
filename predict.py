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

def load_image(path):
    img = Image.open(path).convert("RGB")  # Convert to RGBgrayscale
    img = img.resize((64, 64))  # Resize to 64x64
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = img_array.transpose(2,0,1)
    img_array = img_array[np.newaxis]  # Add batch and channel dimensions
    return img_array

def enable_winograd(model):
    model.conv1.algorithm = "im2col"
    model.conv2.algorithm = "im2col"
    model.conv3.algorithm = "im2col"

    # model.conv1.wino_ready = False
    # model.conv2.wino_ready = False
    # model.conv3.wino_ready = False

def predict(model, image):
    model.eval()
    output = model.forward(image)
    return np.argmax(output)

if __name__ == "__main__":
    model = MyAI(num_classes=13)

    load_model(model, "model.npz")
    enable_winograd(model)

    img = load_image("image.png")
    pred_class = predict(model, img)

    print(f"Predicted class: {pred_class}")
            