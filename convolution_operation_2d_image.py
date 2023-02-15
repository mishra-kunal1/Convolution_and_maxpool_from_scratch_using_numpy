import numpy as np

def convolution_operation_2D_Image(input_image, kernel, stride, pad):
    """
    Performs a 2D convolution operation on a given input_image with a given kernel.

    Args:
    input_image (numpy array): a 2D array representing the input image
    kernel (numpy array): a 2D array representing the weights used for the convolution
    stride (int): the stride used for the convolution operation
    pad (int): the amount of zero padding to be added to the input image

    Returns:
    final_output (numpy array): a 2D array representing the result of the convolution operation
    """
    
    # Get the height and width of the input image and kernel
    input_height, input_width = input_image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Add zero padding to the input image based on the given pad value
    padded_image = np.pad(input_image, pad, 'constant', constant_values=(0, 0))
    
    # Calculate the output height and width based on the input size, kernel size, stride, and pad
    output_height = int((input_height - kernel_height + 2 * pad) / stride) + 1
    output_width = int((input_width - kernel_width + 2 * pad) / stride) + 1
    
    # Create an empty array for the final output
    final_output = np.zeros((output_height, output_width))
    
    # Loop through each element of the final output array
    for h in range(output_height):
        h_start = h * stride
        h_end = h_start + kernel_height
        
        for w in range(output_width):
            w_start = w * stride
            w_end = w_start + kernel_width
            
            # Get the image patch corresponding to the current output element
            image_patch = padded_image[h_start:h_end, w_start:w_end]
            
            # Perform a convolution step on the image patch and the kernel
            #element wise multiplication of two similar sized matrix and taking element wise sum of resultant matrix
            final_output[h, w] = np.sum(np.multiply(image_patch, kernel))
    
    # Return the final output array
    return final_output

