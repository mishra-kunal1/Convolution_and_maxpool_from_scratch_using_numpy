import numpy as np

def max_pool_batch_3D_images(input_images_batch,filter_size,stride):
     
    batch_size, input_height, input_width, input_channels = input_images_batch.shape
    

    # Compute the output image dimensions
    output_height = int(1 + (input_height - filter_size) / stride)
    output_width = int(1 + (input_width - filter_size) / stride)
    
    # Initialize output matrix A
    final_output = np.zeros((batch_size, output_height, output_width, input_channels))              
    # Loop through each input image in the batch
    for index in range(batch_size):
        # Loop through each pixel in the output image
        for h in range(output_height):
            h_start = h * stride
            h_end = h_start + filter_size
            for w in range(output_width):
                w_start = w * stride
                w_end = w_start + filter_size
                # Loop through each channel in the kernel
                for c in range(input_channels):
                    
                    #Extract the image patch and apply pooling operation
                    #we are also apply np.mean() to get average pooling
                    image_patch = final_output[index,h_start:h_end,w_start:w_end,c]
                    final_output[index, h, w, c] = np.max(image_patch)
                    
    return final_output