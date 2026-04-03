import numpy as np

def convolution_forward(layer: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    input_height, input_width = layer.shape
    kernel_height, kernel_width = kernel.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    if not bias.shape == (output_height, output_width):
        raise ValueError("Bias shape must match the output shape.")
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            chosen_region = layer[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(chosen_region * kernel) + bias[i, j]
    
    return output
