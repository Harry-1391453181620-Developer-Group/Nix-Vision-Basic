import numpy as np

class ConvolutionLayer:
    def __init__(self, layer_x: int, layer_y: int, kernel_x: int, kernel_y: int, layer_data: np.ndarray, kernel_data: np.ndarray, bias_data: np.ndarray):
        self.layer_x = layer_x
        self.layer_y = layer_y
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y

        self.layer_data = layer_data
        self.kernel_data = kernel_data
        self.bias_data = bias_data

        self.kernel_gradient = np.zeros_like(kernel_data)
        self.bias_gradient = np.zeros_like(bias_data)

        # For Momentum
        self.kernel_valocity = np.zeros_like(kernel_data)
        self.bias_valocity = np.zeros_like(bias_data)

        self.output: np.ndarray | None = None
        self.input_cache = None
        self.output_cache = None

    def convolution_forward(self) -> np.ndarray:
        output_x = self.layer_x - self.kernel_x + 1
        output_y = self.layer_y - self.kernel_y + 1
        if not self.bias_data.shape == (output_x, output_y):
            raise ValueError("Bias shape must match the output shape.")
        output = np.zeros((output_x, output_y))

        for i in range(output_x):
            for j in range(output_y):
                chosen_region = self.layer_data[i:i+self.kernel_x, j:j+self.kernel_y]
                output[i, j] = np.sum(chosen_region * self.kernel_data) + self.bias_data[i, j]
        
        self.output = output
        return output
    
    def convolution_backward(self, output_gradient: np.ndarray):
        output_x, output_y = output_gradient.shape

        self.kernel_gradient.fill(0)
        self.bias_gradient.fill(0)

        for i in range(output_x):
            for j in range(output_y):
                chosen_region = self.layer_data[i:i+self.kernel_x, j:j+self.kernel_y]
                self.kernel_gradient += chosen_region * output_gradient[i, j]
                self.bias_gradient[i, j] = output_gradient[i, j]

    def momentum_update(self, learning_rate: float, momentum_factor: float):
        self.kernel_velocity = momentum_factor * self.kernel_velocity - learning_rate * self.kernel_gradient
        self.bias_velocity = momentum_factor * self.bias_velocity - learning_rate * self.bias_gradient

        self.kernel_data += self.kernel_velocity
        self.bias_data += self.bias_velocity

class ReLULayer:
    def __init__(self):
        self.input_data: np.ndarray | None = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        output = np.maximum(0, input_data)
        return output
    
    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        relu_gradient = (self.input_data > 0).astype(float)
        gradient_input = gradient_output * relu_gradient
        return gradient_input
    
class FullyConnectedLayer: 
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

        self.input_data = None

        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)

        self.weights_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias)

    def foward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        return input_data @ self.weights + self.bias

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # ∂E/∂W
        self.weights_gradient = self.input_data.T @ output_gradient
        # ∂E/∂b
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        # ∂E/∂X
        input_gradient = output_gradient @ self.weights.T
        return input_gradient
    
    def momentum_update(self, learning_rate: float, momentum: float = 0.9):
        self.weights_velocity = momentum * self.weights_velocity - learning_rate * self.weights_gradient
        self.bias_velocity = momentum * self.bias_velocity - learning_rate * self.bias_gradient

        self.weights += self.weights_velocity
        self.bias += self.bias_velocity