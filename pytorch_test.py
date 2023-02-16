import torch
import torch.nn as nn
import numpy as np
import batch_convolution 


np.random.seed(1)
# Generate random data
## 4 random RGB images of size 9x9x3
input_image_batch = np.random.rand(4, 9, 9, 3).astype(np.float32)
kernel = np.random.rand(8, 5,5 ,3).astype(np.float32)


# Apply custom convolution_operation_batch_3D_images
output_custom =batch_convolution.convolution_operation_batch_3D_images(input_image_batch,kernel,stride=1, pad=2)
#in pytorch the shape of output shape is num_batch x num_channels x height x width
output_custom=output_custom.transpose(0,3,1,2)
print('Output shape of custom convolution')
print(output_custom.shape)



#applying pytorch conv2d layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2, bias=False)
with torch.no_grad():
  conv_layer.weight.data = torch.tensor(kernel.transpose(0, 3, 1, 2))
output_torch=conv_layer(torch.tensor(input_image_batch.transpose(0,3,1,2)))
output_torch=output_torch.detach().numpy()

print(output_torch.shape)
# Compare outputs
print('*'*50)
assert np.allclose(np.round(output_torch,2), np.round(output_custom,2), rtol=1e-5, atol=1e-8)
print("Outputs of both methods are the same")
print('*'*50)
