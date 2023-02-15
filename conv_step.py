def conv_step(input_part, kernel):
    """
    Performs a single convolution step by multiplying an input_part of an input volume with a kernel.

    Args:
    input_part (numpy array): a 2D array representing a subset of the input volume
    kernel (numpy array): a 2D array representing the weights used for the convolution

    Returns:
    result (float): the scalar result of performing the convolution step
    """

    # Perform element-wise multiplication between input_part and kernel
    element_wise_product = np.multiply(input_part, kernel)

    # Sum the results of the element-wise multiplication to obtain a scalar value
    result = np.sum(element_wise_product)

    # Return the scalar result
    return result