import layers


class MyAI:
    def __init__(self, num_classes: int):
        """
        num_classes: 分类类别数量（必须和dataset一致）
        """
        # CNN结构
        self.conv = layers.ConvolutionLayer(3, 3)
        self.relu = layers.ReLULayer()
        self.pool = layers.MaxPoolingLayer(2)

        self.flatten = layers.FlattenLayer()

        # Fully Connected（Lazy Init）
        self.fc = layers.FullyConnectedLayer(None, num_classes)

        self.softmax = layers.SoftmaxLayer()

    def forward(self, x):
        """
        x: (H, W)
        """
        x = self.conv.forward(x)      # -> (H-2, W-2)
        x = self.relu.forward(x)
        x = self.pool.forward(x)      # -> downsample

        x = self.flatten.forward(x)   # -> (1, N)
        x = self.fc.forward(x)        # -> (1, num_classes)
        x = self.softmax.forward(x)

        return x

    def backward(self, grad):
        """
        grad: (1, num_classes)
        """
        grad = self.softmax.backward(grad)
        grad = self.fc.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.conv.backward(grad)

    def update(self, learning_rate: float, momentum: float = 0.9):
        """
        更新参数（目前只有conv + fc有参数）
        """
        self.conv.momentum_update(learning_rate, momentum)
        self.fc.momentum_update(learning_rate, momentum)