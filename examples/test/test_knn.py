import torch

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

point_cloud = torch.rand(1, 3, 20)

idx = knn(point_cloud, k=2)   # (batch_size, num_points, k)
print(idx)
