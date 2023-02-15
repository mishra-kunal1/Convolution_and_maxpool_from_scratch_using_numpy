import tensorflow as tf
import numpy as np
import convolution_operation_batch_3D_images

from tensorflow.keras.layers import Conv2D

# Generate random data
## 4 random RGB images of size 9x9x3
input_image_batch = np.random.randn(4, 9, 9, 3).astype(np.float32)
kernel = np.random.randn(8, 5,5 ,3).astype(np.float32)

# Apply convolution_operation_batch_3D_images

output_custom =convolution_operation_batch_3D_images(input_image_batch,kernel,stride=1, pad=2)
print('Output shape of custom convolution')
print(output_custom.shape)


# Apply TensorFlow's Conv2D layer
init=tf.constant_initializer(kernel.transpose(1,2,3,0))

conv_layer = Conv2D(filters=8, kernel_size=5, strides=1, padding='same', use_bias=False,kernel_initializer=init)

output_tensorflow = conv_layer(tf.constant(input_image_batch))
output_tensorflow = output_tensorflow.numpy()
print('Output shape of tensorflow convolution')
print(output_tensorflow.shape)
# Compare outputs
print('*'*50)
assert np.allclose(np.round(output_tensorflow,2), np.round(output_custom,2), rtol=1e-5, atol=1e-8)
print("Outputs of both methods are the same")
print('*'*50)