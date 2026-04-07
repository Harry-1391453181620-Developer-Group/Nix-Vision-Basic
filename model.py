import layers


class MyAI:
    def __init__(self, num_classes: int):
        # CNN Structure
        self.conv = layers.ConvolutionLayer(num_kernels=8, kernel_x=3, kernel_y=3)
        self.relu = layers.ReLULayer()
        self.pool = layers.MaxPoolingLayer(2)

        self.flatten = layers.FlattenLayer()

        # Fully Connected（Lazy Init）
        self.fc = layers.FullyConnectedLayer(None, num_classes)

        self.softmax = layers.SoftmaxLayer()

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.relu.forward(x)
        x = self.pool.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        x = self.softmax.forward(x)

        return x

    def backward(self, grad):
        grad = self.softmax.backward(grad)
        grad = self.fc.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.conv.backward(grad)

    def update(self, learning_rate: float, momentum: float = 0.9):
        self.conv.momentum_update(learning_rate, momentum)
        self.fc.momentum_update(learning_rate, momentum)