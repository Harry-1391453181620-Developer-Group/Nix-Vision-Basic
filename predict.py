import numpy as np
from PIL import Image
from model import MyAI
from layers import ConvolutionLayer

def load_image(path):
    img = Image.open(path).convert("L")  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to 64x64
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

def enable_winograd(model):
    model.conv1.algorithm = "winograd"
    model.conv2.algorithm = "winograd"

    model.conv1.wino_ready = False
    model.conv2.wino_ready = False

def predict(model, image):
    output = model.forward(image)
    return np.argmax(output)

if __name__ == "__main__":
    model = MyAI(num_classes=13)

    model.load_model("model.npz")
    enable_winograd(model)

    img = load_image("image.png")
    pred_class = model.predict(img)

    pred = np.argmax(pred_class)
    print(f"Predicted class: {pred}")
            