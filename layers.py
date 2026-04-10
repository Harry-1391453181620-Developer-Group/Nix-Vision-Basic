import numpy as np

class ConvolutionLayer:
    def __init__(self, num_kernels: int, kernel_x: int, kernel_y: int, num_channels: int=1):
        self.num_kernels = num_kernels
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.num_channels = num_channels

        self.kernel_data = np.random.randn(num_kernels, num_channels, kernel_x, kernel_y) * 0.1
        self.bias_data = np.zeros((num_kernels,))

        self.kernel_gradient = np.zeros_like(self.kernel_data)
        self.bias_gradient = np.zeros_like(self.bias_data)

        # For Momentum
        self.kernel_velocity = np.zeros_like(self.kernel_data)
        self.bias_velocity = np.zeros_like(self.bias_data)

        self.input_cache = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, :, :]

        self.input_cache = input_data

        input_c, input_x, input_y = input_data.shape
        
        # Security check
        if input_x < self.kernel_x or input_y < self.kernel_y:
            raise ValueError("Kernel size larger than input.")
        output_x = input_x - self.kernel_x + 1
        output_y = input_y - self.kernel_y + 1

        output = np.zeros((self.num_kernels, output_x, output_y))
        for k in range(self.num_kernels):
            for i in range(output_x):
                for j in range(output_y):
                    region = self.input_cache[:, i:i+self.kernel_x, j:j+self.kernel_y]
                    output[k, i, j] = np.sum(region * self.kernel_data[k]) + self.bias_data[k]
        
        return output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Security check
        if self.input_cache is None:
            raise ValueError("Run forward before backward.")
        input_data = self.input_cache

        K, output_x, output_y = output_gradient.shape

        self.kernel_gradient.fill(0)
        self.bias_gradient.fill(0)

        input_gradient = np.zeros_like(input_data)
    
        for k in range(K):
            for i in range(output_x):
                for j in range(output_y):
                    region = input_data[:, i:i+self.kernel_x, j:j+self.kernel_y]
                    self.kernel_gradient[k] += region * output_gradient[k, i, j]
                    self.bias_gradient[k] += output_gradient[k, i, j]
                    input_gradient[:, i:i + self.kernel_x, j:j + self.kernel_y] += \
                        self.kernel_data[k] * output_gradient[k, i, j]

        #self.kernel_gradient /= (output_x * output_y)
        #self.bias_gradient /= (output_x * output_y)
        return input_gradient

    def momentum_update(self, learning_rate: float, momentum: float = 0.9):
        self.kernel_velocity = momentum * self.kernel_velocity - learning_rate * self.kernel_gradient
        self.bias_velocity = momentum * self.bias_velocity - learning_rate * self.bias_gradient

        self.kernel_data += self.kernel_velocity
        self.bias_data += self.bias_velocity

class ReLULayer:
    def __init__(self):
        self.input_cache = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_cache = input_data
        output = np.maximum(0, input_data)
        return output
    
    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        return gradient_output * (self.input_cache > 0).astype(float)
    
class FullyConnectedLayer: 
    def __init__(self, input_size: int | None, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        # Lazy Init
        if input_size is not None:
            self._initialize_parameters(input_size)
        else:
            self.weights = None
            self.bias = None
            self.weights_gradient = None
            self.bias_gradient = None
            self.weights_velocity = None
            self.bias_velocity = None

        self.input_cache = None
        self.original_shape = None

    def _initialize_parameters(self, input_size: int):
        self.weights = np.random.randn(input_size, self.output_size) * 0.1
        self.bias = np.zeros((1, self.output_size))

        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)

        # For Momentum
        self.weights_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.original_shape = input_data.shape

        #Flatten
        if input_data.ndim != 2:
            input_data = input_data.reshape(1, -1)
        
        self.input_cache = input_data
        if self.weights is None:
            input_size = input_data.shape[1]
            self._initialize_parameters(input_size)

        return input_data @ self.weights + self.bias

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Protect backward from lazy init
        if self.weights is None:
            raise ValueError("Layer not initialized. Run forward first.")
        # ∂E/∂W
        self.weights_gradient = self.input_cache.T @ output_gradient
        # ∂E/∂b
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Average gradients over batch
        batch_size = output_gradient.shape[0]
        self.weights_gradient /= batch_size
        self.bias_gradient /= batch_size
        
        # ∂E/∂X
        input_gradient = output_gradient @ self.weights.T
        return input_gradient
    
    def momentum_update(self, learning_rate: float, momentum: float = 0.9):
        self.weights_velocity = momentum * self.weights_velocity - learning_rate * self.weights_gradient
        self.bias_velocity = momentum * self.bias_velocity - learning_rate * self.bias_gradient

        self.weights += self.weights_velocity
        self.bias += self.bias_velocity

class MaxPoolingLayer:
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.input_cache = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_cache = input_data

        input_z, input_x, input_y = input_data.shape
        output_x = input_x // self.pool_size
        output_y = input_y // self.pool_size

        output = np.zeros((input_z, output_x, output_y))

        for k in range(input_z):
            for i in range(output_x):
                for j in range(output_y):
                    region = input_data[
                        k,
                        i * self.pool_size: (i + 1) * self.pool_size,
                        j * self.pool_size: (j + 1) * self.pool_size
                    ]
                    output[k, i, j] = np.max(region)
        return output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        input_data = self.input_cache
        input_z, input_x, input_y = input_data.shape
        output_z, output_x, output_y = output_gradient.shape
        input_gradient = np.zeros_like(input_data)

        for k in range(input_z):
            for i in range(output_x):
                for j in range(output_y):
                    region = input_data[
                        k,
                        i * self.pool_size: (i + 1) * self.pool_size,
                        j * self.pool_size: (j + 1) * self.pool_size
                    ]
                    max_index = np.unravel_index(np.argmax(region), region.shape)

                    input_gradient[
                        k,
                        i * self.pool_size + max_index[0],
                        j * self.pool_size + max_index[1]
                    ] = output_gradient[k, i, j]
        return input_gradient
    
class FlattenLayer:
    def __init__(self):
        self.original_shape = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.original_shape = input_data.shape
        return input_data.reshape(1, -1)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient.reshape(self.original_shape)
    
class SoftmaxLayer:
    def __init__(self):
        self.output_cache = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_cache = probabilities
        return probabilities
    
    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        return gradient_output
    
class CrossEntropyLossLayer:
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        epsilon = 1e-12
        prediction = np.clip(prediction, epsilon, 1. - epsilon)
        loss = -np.mean(np.sum(target * np.log(prediction), axis=1))
        return loss
    
    def backward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return prediction - target