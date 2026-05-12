import numpy as np

class ConvolutionLayer:
    def __init__(self, num_kernels: int, kernel_x: int, kernel_y: int, num_channels: int=1, algorithm: str="im2col"):
        self.num_kernels = num_kernels
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.num_channels = num_channels

        self.kernel_data = (
            np.random.randn(num_kernels, num_channels, kernel_x, kernel_y).astype(np.float32)
            * np.float32(np.sqrt(2.0 / (num_channels * kernel_x * kernel_y)))
        )
        self.bias_data = np.zeros((num_kernels,), dtype=np.float32)

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
        #self.wino_ready = False

        # train/eval mode flag
        self.training = True

    # GEMM (im2col)
    def _forward_im2col(self, input_data: np.ndarray) -> np.ndarray:
        """Packed im2col forward pass, ready to be used"""
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, np.newaxis, :, :]
        elif input_data.ndim == 3:
            input_data = input_data[np.newaxis, :, :, :]

        input_batch_num, input_channels, input_x, input_y = input_data.shape

        if input_x < self.kernel_x or input_y < self.kernel_y:
            raise ValueError("Kernel size larger than input.")
        output_x = input_x - self.kernel_x + 1
        output_y = input_y - self.kernel_y + 1

        #col (flattened array)
        col = np.zeros((input_batch_num * output_x * output_y, input_channels * self.kernel_x * self.kernel_y), dtype=np.float32) #Using zeros to setup
        row_index = 0
        for b in range(input_batch_num):
            for i in range(output_x):
                for j in range(output_y):
                    region = input_data[b, :, i:i+self.kernel_x, j:j+self.kernel_y] # the region from original input that should be flattened 
                    flattened_region = region.flatten()
                    col[row_index] = flattened_region
                    row_index += 1

        flattened_kernel = self.kernel_data.reshape(self.num_kernels, -1)

        output = (col @ flattened_kernel.T)
        output = output.reshape(input_batch_num, output_x, output_y, self.num_kernels)
        output = output.transpose(0, 3, 1, 2)
        output += self.bias_data.reshape(1, -1, 1, 1)
        self.col_cache = col
        self.input_cache = input_data
        return output
    
    def _backward_col2im(self, output_gradient: np.ndarray) -> np.ndarray:
        """Packed col2im backward pass, ready to be used"""
        B, K, output_x, output_y = output_gradient.shape
        
        gradient_reshaped = output_gradient.transpose(0,2,3,1)
        gradient_reshaped = gradient_reshaped.reshape(-1, self.num_kernels)
        self.kernel_gradient = (gradient_reshaped.T @ self.col_cache).reshape(self.kernel_data.shape) / B
        self.bias_gradient = np.sum(gradient_reshaped, axis=0) / B

        kernel_matrix = self.kernel_data.reshape(self.num_kernels, -1)
        col_gradient = gradient_reshaped @ kernel_matrix

        B, input_channels, input_x, input_y = self.input_cache.shape
        input_gradient = np.zeros((B, input_channels, input_x, input_y), dtype=output_gradient.dtype)

        row_index = 0
        for b in range(B):
            for i in range(output_x):
                for j in range(output_y):
                    # reshape one row back
                    patch = col_gradient[row_index].reshape(input_channels, self.kernel_x, self.kernel_y)
                    #add reshaped stuff to corresponding region
                    input_gradient[b, :, i:i+self.kernel_x, j:j+self.kernel_y] += patch
                    row_index += 1

        return input_gradient
    
#    # Winograd IS DEPRECIATED CURRENTLY, BUT THERE IS A CHANCE FOR IT TO BE USED AGAIN. This is because the current CNN structure is not suitable for Winograd.
#    def _init_winograd_matrices(self):
#        # Winograd F(2x2, 3x3) matrices
#        self.G = np.array([[1, 0, 0],
#                           [0.5, 0.5, 0.5],
#                           [0.5, -0.5, 0.5],
#                           [0, 0, 1]])
#
#        self.B = np.array([[1, 0, -1, 0],
#                           [0, 1, 1, 0],
#                           [0, -1, 1, 0],
#                           [0, 1, 0, -1]])
#
#        self.A = np.array([[1, 1, 1, 0],
#                           [0, 1, -1, -1]])
#        
#    def _transform_kernels_winograd(self):
#        G = self.G
#        GT = G.T
#
#        self.U = np.zeros((self.num_kernels, self.num_channels, 4, 4))
#
#        for k in range(self.num_kernels):
#            for c in range(self.num_channels):
#                self.U[k, c] = G @ self.kernel_data[k, c] @ GT
#
#        self.wino_ready = True
#
#    def _forward_winograd(self, input_data):
#        """Packed winograd forward pass, ready to be used VECTORIZED TO INCREASE SPEED"""
#        if input_data.ndim == 2:
#            input_data = input_data[np.newaxis, np.newaxis, :, :]
#        elif input_data.ndim == 3:
#            input_data = input_data[np.newaxis, :, :, :]
#
#        if not self.wino_ready:
#            self._transform_kernels_winograd()
#        B, C, H, W = input_data.shape
#
#        if (H - 2) % 2 != 0 or (W - 2) % 2 != 0:  # Only supports kernel=3, stride=1, no padding currently.
#            return self._forward_im2col(input_data)  # Fallback to im2col if dimensions are not suitable for Winograd
#
#        if self.kernel_x != 3 or self.kernel_y != 3:
#            return self._forward_im2col(input_data)
#
#        self.input_cache = input_data
#
#        out_x = H - self.kernel_x + 1
#        out_y = W - self.kernel_y + 1
#
#        max_i = H - 4 + 1
#        max_j = W - 4 + 1
#
#        xs = np.arange(0, max_i, 2)
#        ys = np.arange(0, max_j, 2)
#
#        T = len(xs) * len(ys)
#
#        tiles = np.empty((B, T, C, 4, 4), dtype=input_data.dtype)
#        t = 0
#        for i in xs:
#            for j in ys:
#                tiles[:, t] = input_data[:, :, i:i+4, j:j+4]
#                t += 1
#        V = np.einsum('ab,btcbj->btcaj', self.B, tiles) # left * B
#        V = np.einsum('btcai,ij->btcaj', V, self.B.T) # right * B.T
#
#        M = np.einsum('kcij,btcij->btkij', self.U, V) # U * V
#        Y = np.einsum('ab,btkbj->btkaj', self.A, M)
#        Y = np.einsum('btkai,ij->btkaj', Y, self.A.T)
#
#        output = np.zeros((B, self.num_kernels, out_x, out_y), dtype=input_data.dtype)
#
#        t = 0
#        for i in xs:
#            for j in ys:
#                output[:, :, i:i+2, j:j+2] = Y[:, t] + self.bias_data[:, None, None]
#                t += 1
#
#        return output

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Only use packed methods"""
        #use_winograd = (
        #    self.kernel_x == 3 and
        #    self.kernel_y == 3 and 
        #    self.algorithm in ["auto", "winograd"]
        #)

        #if use_winograd and not self.wino_ready:
        #    self._init_winograd_matrices()
        #    self._transform_kernels_winograd()
#
        #if use_winograd:
        #    return self._forward_winograd(input_data)
#
        return self._forward_im2col(input_data)
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Only col2im"""
        return self._backward_col2im(output_gradient)

    def momentum_update(self, learning_rate: float, momentum: float = 0.9, l2_lambda: float = 0.0001):
        #if hasattr(self, 'U'):
        #    del self.U
        
        self.kernel_gradient += l2_lambda * self.kernel_data

        self.kernel_velocity = momentum * self.kernel_velocity - learning_rate * self.kernel_gradient
        self.bias_velocity = momentum * self.bias_velocity - learning_rate * self.bias_gradient

        self.kernel_data += self.kernel_velocity
        self.bias_data += self.bias_velocity

        #self.wino_ready = False

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
        self.weights = (
            np.random.randn(input_size, self.output_size).astype(np.float32)
            * np.float32(np.sqrt(2.0 / input_size))
        )
        self.bias = np.zeros((1, self.output_size), dtype=np.float32)

        self.weights_gradient = np.zeros_like(self.weights, dtype=np.float32)
        self.bias_gradient = np.zeros_like(self.bias, dtype=np.float32)

        # For Momentum
        self.weights_velocity = np.zeros_like(self.weights, dtype=np.float32)
        self.bias_velocity = np.zeros_like(self.bias, dtype=np.float32)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.original_shape = input_data.shape

        #Flatten
        if input_data.ndim != 2:
            input_data = input_data.reshape(input_data.shape[0], -1)
        
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

        B, input_z, input_x, input_y = input_data.shape
        output_x = input_x // self.pool_size
        output_y = input_y // self.pool_size

        output = np.zeros((B, input_z, output_x, output_y))
        for b in range(B):
            for k in range(input_z):
                for i in range(output_x):
                    for j in range(output_y):
                        region = input_data[
                            b,
                            k,
                            i * self.pool_size: (i + 1) * self.pool_size,
                            j * self.pool_size: (j + 1) * self.pool_size
                        ]   
                        output[b, k, i, j] = np.max(region)
        return output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        input_data = self.input_cache
        B, input_z, input_x, input_y = input_data.shape
        B, output_z, output_x, output_y = output_gradient.shape
        input_gradient = np.zeros_like(input_data)

        for b in range(B):
            for k in range(input_z):
                for i in range(output_x):
                    for j in range(output_y):
                        region = input_data[
                            b,
                            k,
                            i * self.pool_size: (i + 1) * self.pool_size,
                            j * self.pool_size: (j + 1) * self.pool_size
                        ]
                        max_index = np.unravel_index(np.argmax(region), region.shape)

                        input_gradient[
                            b,
                            k,
                            i * self.pool_size + max_index[0],
                            j * self.pool_size + max_index[1]
                        ] += output_gradient[b, k, i, j]  # += for future extentions.
        return input_gradient

# Depreciated FlattenLayer, not used, replaced by GlobalAvgPoolingLayer
#class FlattenLayer:
#    """DEPRECIATED"""
#    def __init__(self):
#        self.original_shape = None
#
#    def forward(self, input_data: np.ndarray) -> np.ndarray:
#        self.original_shape = input_data.shape
#        return input_data.reshape(1, -1)
#
#    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
#        return output_gradient.reshape(self.original_shape)
    
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
        return (prediction - target)
    
class GlobalAvgPoolingLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return input_data.mean(axis=(2, 3))
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        B, C, H, W = self.input_shape
        return output_gradient.reshape(B, C, 1, 1) * \
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