import spconv.pytorch as spconv
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# an example of how to use spconv SubMConv3d on a set of points
# features = # your features with shape [N, num_channels]
# indices = # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
# spatial_shape = # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
# batch_size = # batch size of your sparse tensor.
# x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
features = torch.tensor([[0, 0, 5.0], [0, 0, 6.0], [0, 0, 7.0]])
features = features.to(device)

batch_size = 1
batch_size = torch.tensor(batch_size)


indices = torch.tensor([[0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 2]], dtype=torch.int32)
indices = indices.int()
indices = indices.to(device)
# spatial_shape[i] is shape of indices[:, 1 + i]
# ndim = indices.shape[1] - 1
spatial_shape = np.array([1, 3], dtype=np.int32)
spatial_shape = torch.from_numpy(spatial_shape)
spatial_shape = spatial_shape.to(device)

conv3d = spconv.SubMConv3d(3, 4, kernel_size=1, stride=1, padding=0, bias=False, indice_key="subm0").to(device)

x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
y = conv3d(x)
print(y.features)


