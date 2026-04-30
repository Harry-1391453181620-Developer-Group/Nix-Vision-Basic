import numpy as np

class ConvolutionLayer:
    def __init__(self, num_kernels: int, kernel_x: int, kernel_y: int, num_channels: int=1, algorithm: str="auto"):
        self.num_kernels = num_kernels
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.num_channels = num_channels

        self.kernel_data = np.random.randn(num_kernels, num_channels, kernel_x, kernel_y) \
                           * np.sqrt(2.0 / (num_channels * kernel_x * kernel_y)) 
        self.bias_data = np.zeros((num_kernels,))

        self.kernel_gradient = np.zeros_like(self.kernel_data)
        self.bias_gradient = np.zeros_like(self.bias_data)

        # For Momentum
        self.kernel_velocity = np.zeros_like(self.kernel_data)
        self.bias_velocity = np.zeros_like(self.bias_data)

        self.input_cache = None
        self.col_cache = None # col cache for backward

        # Convolution algorithm
        self.algorithm = algorithm

        # Winograd cache flag
        self.wino_ready = False

        # train/eval mode flag
        self.training = True

    # GEMM (im2col)
    def _forward_im2col(self, input_data: np.ndarray) -> np.ndarray:
        """Packed im2col forward pass, ready to be used"""
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, :, :]
        
        self.input_cache = input_data

        input_channels, input_x, input_y = input_data.shape

        if input_x < self.kernel_x or input_y < self.kernel_y:
            raise ValueError("Kernel size larger than input.")
        output_x = input_x - self.kernel_x + 1
        output_y = input_y - self.kernel_y + 1

        #col (flattened array)
        col = np.zeros((output_x * output_y, input_channels * self.kernel_x * self.kernel_y)) #Using zeros to setup
        row_index = 0
        for i in range(output_x):
            for j in range(output_y):
                region = input_data[:, i:i+self.kernel_x, j:j+self.kernel_y] # the region from original input that should be flattened 
                flattened_region = region.flatten()
                col[row_index] = flattened_region
                row_index += 1

        flattened_kernel = self.kernel_data.reshape(self.num_kernels, -1)

        output = (flattened_kernel @ col.T) + self.bias_data.reshape(-1, 1)
        output = output.reshape(self.num_kernels, output_x, output_y)
        self.col_cache = col
        return output
    
    def _backward_col2im(self, output_gradient: np.ndarray) -> np.ndarray:
        """Packed col2im backward pass, ready to be used"""
        K, output_x, output_y = output_gradient.shape
        
        gradient_reshaped = output_gradient.reshape(self.num_kernels, -1)
        self.kernel_gradient = (gradient_reshaped @ self.col_cache).reshape(self.kernel_data.shape)
        self.bias_gradient = np.sum(gradient_reshaped, axis=1)

        kernel_matrix = self.kernel_data.reshape(self.num_kernels, -1)
        col_gradient = (kernel_matrix.T @ gradient_reshaped).T

        input_channels, input_x, input_y = self.input_cache.shape
        input_gradient = np.zeros((input_channels, input_x, input_y))

        row_index = 0
        for i in range(output_x):
            for j in range(output_y):
                # reshape one row back
                patch = col_gradient[row_index].reshape(input_channels, self.kernel_x, self.kernel_y)
                #add reshaped stuff to corresponding region
                input_gradient[:, i:i+self.kernel_x, j:j+self.kernel_y] += patch
                row_index += 1

        return input_gradient
    
    # Winograd
    def _init_winograd_matrices(self):
        # Winograd F(2x2, 3x3) matrices
        self.G = np.array([[1, 0, 0],
                           [0.5, 0.5, 0.5],
                           [0.5, -0.5, 0.5],
                           [0, 0, 1]])

        self.B = np.array([[1, 0, -1, 0],
                           [0, 1, 1, 0],
                           [0, -1, 1, 0],
                           [0, 1, 0, -1]])

        self.A = np.array([[1, 1, 1, 0],
                           [0, 1, -1, -1]])
        
    def _transform_kernels_winograd(self):
        G = self.G
        GT = G.T

        self.U = np.zeros((self.num_kernels, self.num_channels, 4, 4))

        for k in range(self.num_kernels):
            for c in range(self.num_channels):
                self.U[k, c] = G @ self.kernel_data[k, c] @ GT

        self.wino_ready = True

    def _forward_winograd(self, input_data):
        """Packed winograd forward pass, ready to be used VECTORIZED TO INCREASE SPEED"""
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, :, :]

        C, H, W = input_data.shape

        if (H - 2) % 2 != 0 or (W - 2) % 2 != 0:
            return self._forward_im2col(input_data)  # Fallback to im2col if dimensions are not suitable for Winograd

        self.input_cache = input_data
        input_channels, input_x, input_y = input_data.shape
        output_x = input_x - self.kernel_x + 1
        output_y = input_y - self.kernel_y + 1

        col = np.zeros((output_x * output_y, input_channels * self.kernel_x * self.kernel_y))
        row_index = 0
        for i in range(output_x):
            for j in range(output_y):
                region = input_data[:, i:i+self.kernel_x, j:j+self.kernel_y]
                col[row_index] = region.flatten()
                row_index += 1

        self.col_cache = col

        out_x = H - 2
        out_y = W - 2

        xs = np.arange(0, out_x, 2)
        ys = np.arange(0, out_y, 2)
        T = len(xs) * len(ys)

        tiles = np.empty((T, C, 4, 4), dtype=input_data.dtype)
        t = 0
        for i in xs:
            for j in ys:
                tiles[t] = input_data[:, i:i+4, j:j+4]
                t += 1
        V = np.einsum('ab,tcbj->tcaj', self.B, tiles) # left * B
        V = np.einsum('tcai,ij->tcaj', V, self.B.T) # right * B.T

        M = np.einsum('kcij,tcij->tkij', self.U, V) # U * V
        Y = np.einsum('ab,tkbj->tkaj', self.A, M)
        Y = np.einsum('tkai,ij->tkaj', Y, self.A.T)

        output = np.zeros((self.num_kernels, out_x, out_y), dtype=input_data.dtype)

        t = 0
        for i in xs:
            for j in ys:
                output[:, i:i+2, j:j+2] = Y[t] + self.bias_data[:, None, None]
                t += 1

        return output

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Only use packed methods"""
        use_winograd = (
            self.kernel_x == 3 and
            self.kernel_y == 3 and 
            self.algorithm in ["auto", "winograd"]
        )

        if use_winograd and not self.wino_ready:
            self._init_winograd_matrices()
            self._transform_kernels_winograd()

        if use_winograd:
            return self._forward_winograd(input_data)

        return self._forward_im2col(input_data)
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Only col2im"""
        return self._backward_col2im(output_gradient)

    def momentum_update(self, learning_rate: float, momentum: float = 0.9, l2_lambda: float = 0.0001):
        if hasattr(self, 'U'):
            del self.U
        
        self.wino_ready = False
        self.kernel_gradient += l2_lambda * self.kernel_data

        self.kernel_velocity = momentum * self.kernel_velocity - learning_rate * self.kernel_gradient
        self.bias_velocity = momentum * self.bias_velocity - learning_rate * self.bias_gradient

        self.kernel_data += self.kernel_velocity
        self.bias_data += self.bias_velocity

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

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
        self.weights = np.random.randn(input_size, self.output_size) \
                       * np.sqrt(2.0 / input_size)
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

        # reshape back to input shape
        input_gradient = input_gradient.reshape(self.original_shape)

        return input_gradient
    
    def momentum_update(self, learning_rate: float, momentum: float = 0.9, l2_lambda: float = 0.0001):
        self.weights_gradient += l2_lambda * self.weights

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

# Depreciated FlattenLayer, not used, replaced by GlobalAvgPoolingLayer
class FlattenLayer:
    """DEPRECIATED"""
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
    
class GlobalAvgPoolingLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return input_data.mean(axis=(1, 2)).reshape(1, -1)
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        C, H, W = self.input_shape
        return output_gradient.reshape(C, 1, 1) * \
               np.ones(self.input_shape) / (H * W)
    

class DropoutLayer:
    def  __init__(self, dropout_prob: float=0.5):
        self.dropout_prob = dropout_prob
        self.mask = None
        self.training = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = (np.random.rand(*input_data.shape) >= self.dropout_prob).astype(float)
            return input_data * self.mask / (1.0 - self.dropout_prob)
        else:
            return input_data
    
    def backward(self, gradient_output: np.ndarray) -> np.ndarray:
        if self.training and self.mask is not None:
            return gradient_output * self.mask / (1.0 - self.dropout_prob)
        else:
            return gradient_output
        
    def train(self):
        self.training = True

    def eval(self):
        self.training = False