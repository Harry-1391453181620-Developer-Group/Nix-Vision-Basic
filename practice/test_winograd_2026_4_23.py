import numpy as np

class WinogradConv2D:
    def __init__(self, input: np.ndarray, kernel: np.ndarray):
        self.input = input
        self.kernel = kernel
        self.output = None

    def forward(self):
        ip = self.input
        k = self.kernel

        G = np.ndarray([
            [1, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0, 0, 1]
        ])

        G_T = G.T

        G_transformed = G @ k @ G_T

        B = np.ndarray([
            [1, 0, -1, 0],
            [0, 1, 1, 0],
            [0, -1, 1, 0],
            [0, 1, 0, -1]
        ])

        B_T = B.T

        B_transformed = B @ ip @ B_T

        multiplyed = B_transformed * G_transformed

        A = np.ndarray([
            [1, 1, 1, 0],
            [0, 1, -1, -1]
        ])

        A_T = A.T

        Y = A @ multiplyed @ A_T

        self.output = Y
        return Y

if __name__ == "__main__":
    input = np.array([[1, 2, 3, 4], 
                      [5, 6, 7, 8], 
                      [9, 10, 11, 12], 
                      [13, 14, 15, 16]])
    
    kernel = np.array([[1, 0], 
                       [0, -1]])
    
    conv = WinogradConv2D(input, kernel)
    output = conv.forward()
    print("Output:\n", output)