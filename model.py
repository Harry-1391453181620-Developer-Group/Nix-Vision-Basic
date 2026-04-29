import layers
import numpy as np


class MyAI:
    def __init__(self, num_classes: int):
        # CNN Structure
        self.conv1 = layers.ConvolutionLayer(num_kernels=16, num_channels=1, kernel_x=3, kernel_y=3, algorithm="im2col")
        self.relu1 = layers.ReLULayer()
        self.pool1 = layers.MaxPoolingLayer(2)

        self.conv2 = layers.ConvolutionLayer(num_kernels=3244, num_channels=8, kernel_x=3, kernel_y=3, algorithm="im2col")
        self.relu2 = layers.ReLULayer()
        self.pool2 = layers.MaxPoolingLayer(2)

        self.gap = layers.GlobalAvgPoolingLayer()

        # Fully Connected（Lazy Init）
        self.fc1 = layers.FullyConnectedLayer(None, 64)
        self.fc_relu = layers.ReLULayer()
        self.fc2 = layers.FullyConnectedLayer(64, num_classes)

        self.softmax = layers.SoftmaxLayer()

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.gap.forward(x)

        x = self.fc1.forward(x)
        x = self.fc_relu.forward(x)
        x = self.fc2.forward(x)

        x = self.softmax.forward(x)

        return x

    def backward(self, grad):
        grad = self.softmax.backward(grad)

        grad = self.fc2.backward(grad)
        grad = self.fc_relu.backward(grad)
        grad = self.fc1.backward(grad)
    

        grad = self.gap.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

    def update(self, learning_rate: float, momentum: float = 0.9, l2_lambda: float = 0.0001):
        self.conv1.momentum_update(learning_rate, momentum, l2_lambda)
        self.conv2.momentum_update(learning_rate, momentum, l2_lambda)
        self.fc1.momentum_update(learning_rate, momentum, l2_lambda)
        self.fc2.momentum_update(learning_rate, momentum, l2_lambda)

    def predict(self, x):
        return self.forward(x)
    
    def load_model(self, path="model.npz"):
        d = np.load(path)
        self.conv1.kernel_data = d["conv1_kernels"]
        self.conv1.bias_data   = d["conv1_bias"]
        self.conv2.kernel_data = d["conv2_kernels"]
        self.conv2.bias_data   = d["conv2_bias"]
        self.fc1.weights = d["fc1_weights"]
        self.fc1.bias    = d["fc1_bias"]
        self.fc2.weights = d["fc2_weights"]
        self.fc2.bias    = d["fc2_bias"]
        print(f"Model loaded from {path}")