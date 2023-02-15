import numpy as np


def convolution_operation_batch_3D_images(input_images_batch, kernel, stride, pad):
    """
    Perform 2D convolution operation on a batch of 3D input images with a given kernel.

    Args:
    input_image_batch (np.array): A 4D numpy array of shape (batch_size, input_height, input_width, input_channels),
                                  where batch_size is the number of input images, input_height is the height of each input
                                  image, input_width is the width of each input image, and input_channels is the number
                                  of channels in each input image.
    kernel (np.array): A 3D numpy array of shape (kernel_height, kernel_width, kernel_channels), where kernel_height is the
                       height of the kernel, kernel_width is the width of the kernel, and kernel_channels is the number of
                       channels in the kernel.
    stride (int): The stride of the convolution operation.
    pad (int): The number of pixels to pad the input image with.

    Returns:
    final_output (np.array): A 4D numpy array of shape (batch_size, output_height, output_width, input_channels), where
                             output_height is the height of each output image and output_width is the width of each output
                             image.
    """

    # Get the shape of the input image and kernel
    batch_size, input_height, input_width, input_channels = input_images_batch.shape
    num_filters, kernel_height, kernel_width, num_channels = kernel.shape

    # Compute the output image dimensions
    output_height = int((input_height - kernel_height + 2 * pad) / stride) + 1
    output_width = int((input_width - kernel_width + 2 * pad) / stride) + 1

    # Pad the input image with zeros
    batch_padded_image = np.pad(input_images_batch, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 
                                mode='constant', constant_values=(0, 0))

    # Create an empty numpy array to hold the output images
    final_output = np.zeros((batch_size, output_height, output_width, num_filters))

    # Loop through each input image in the batch
    for index in range(batch_size):
        current_padded_image = batch_padded_image[index]
        # Loop through each pixel in the output image
        for h in range(output_height):
            h_start = h * stride
            h_end = h_start + kernel_height
            for w in range(output_width):
                w_start = w * stride
                w_end = w_start + kernel_width
                # Loop through each channel in the kernel
                for c in range(num_filters):
                    # Extract the image patch and apply the convolution operation
                    image_patch = current_padded_image[h_start:h_end,w_start:w_end, :]
                    final_output[index, h, w, c] = conv_step(image_patch, kernel[c:c+1,:,:,:])
                    
    return final_output
