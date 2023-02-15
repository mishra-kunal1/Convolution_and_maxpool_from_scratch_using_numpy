# Convolution_and_maxpool_from_scratch_using_numpy

Convolution is an operation commonly used in machine learning that involves "sliding" a small filter (a matrix of numbers) over a larger input matrix and computing a dot product between the two matrices at each position. This process creates a new output matrix that summarizes how the input matrix features match the filter pattern.

Let's understand the Convolution operation on 2d images through this example

![Example GIF](https://miro.medium.com/v2/resize:fit:1070/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)

Now let's write code for it and check the output

```python
import numpy as np
import conv_step

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
            final_output[h, w] = conv_step(image_patch, kernel)
    
    # Return the final output array
    return final_output

```

If we call our function and perform the convolution overation on the above image we get the following ouput

<img width="620" alt="image" src="https://user-images.githubusercontent.com/99056351/219140971-a662bc34-12bb-472d-8452-092e78a63bc7.png">

## Voila, 2d Convolutions are working


    for h in range(output_height):
