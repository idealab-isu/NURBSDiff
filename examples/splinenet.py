import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import h5py
import numpy as np
import geomdl
from scipy.special import comb
from geomdl import BSpline
from geomdl.visualization import VisMPL
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt
torch.manual_seed(0)


EPS = np.finfo(np.float32).eps

#Encoder utils
def knn(x, k):
    batch_size = x.shape[0]
    indices = np.arange(0, k)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        idx = distances.topk(k=k, dim=-1)[1][:, :, indices]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous()
    x = x.view(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx_base = idx_base.cuda(torch.get_device(x))
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DGCNNControlPoints(nn.Module):
    def __init__(self, num_control_points, num_points=40, mode=0):
        """
        Encoder Network
        Control points prediction network. Takes points as input
        and outputs control points grid.
        :param num_control_points: size of the control points grid.
        :param num_points: number of nearest neighbors used in DGCNN.
        :param mode: different modes are used that decides different number of layers.
        """
        super(DGCNNControlPoints, self).__init__()
        self.k = num_points
        self.mode = mode
        if self.mode == 0:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.controlpoints = num_control_points
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024, 1024, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024, 3 * (self.controlpoints ** 2), 1)

            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        if self.mode == 1:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0

            self.conv1 = nn.Sequential(nn.Conv2d(6, 128, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv2 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv3 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv4 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv5 = nn.Sequential(nn.Conv1d(1024 + 128, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.controlpoints = num_control_points
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024, 1024, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024, 3 * (self.controlpoints ** 2), 1)
            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        self.tanh = nn.Tanh()

    def forward(self, x, weights=None):
        """
        :param weights: weights of size B x N
        """
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        if isinstance(weights, torch.Tensor):
            weights = weights.reshape((1, 1, -1))
            x = x * weights

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x1 = torch.unsqueeze(x1, 2)

        x = F.dropout(F.relu(self.bn6(self.conv6(x1))), self.drop)

        x = F.dropout(F.relu(self.bn7(self.conv7(x))), self.drop)
        x = self.conv8(x)
        x = self.tanh(x[:, :, 0])

        x = x.view(batch_size, self.controlpoints * self.controlpoints, 3)
        return x


if __name__ == "__main__":
    path = "closed_splines.h5"

    with h5py.File(path,"r") as hf:
        input_points = np.array(hf.get(name="points")).astype(np.float32)
        input_control_points = np.array(hf.get(name="controlpoints")).astype(np.float32)

    num_epochs = 1000
    batch_size = 8
    learning_rate = 3e-4

    train_data = input_points[0:80]
    train_control_points = input_control_points[0:80]


    

    #Initalize encoder
    encoder = DGCNNControlPoints(20, num_points=10, mode=1)
    encoder.cuda()

    #Initialize decoder
    decoder = SurfEval(20,20, dimension=3, p=2, q=2, out_dim_u=40, out_dim_v=40, method='tc', dvc='cuda')
    decoder.cuda()

    #Initalize Adam optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=learning_rate)

    for epoch in range(num_epochs):
        print('Epoch: ',epoch)
        for i in range(0,len(train_data),batch_size):
            
            # torch.cuda.empty_cache()
            optimizer.zero_grad()

            points_ = train_data[i:i+batch_size]
            control_points_ = train_control_points[i:i+batch_size]
            
            control_points = Variable(torch.from_numpy(control_points_.astype(np.float32))).cuda()
            points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()

            ground_truth_control_points = control_points.clone().detach()
            ground_truth_control_points = ground_truth_control_points.view(-1,400,3)

            # print('Encoder input')
            # print(points.shape)
            points = points.permute(0, 2, 1)
            # print('After permute')
            # print(points.shape)

            encoder_output = encoder(points[:,:,0:700])
            
            # print('Encoder output')
            # print(encoder_output.shape)

            encoder_loss= torch.nn.functional.mse_loss(ground_truth_control_points, encoder_output)
            print('Encoder loss:',encoder_loss)

            #Decoder network

            # ground_truth_decoder = encoder_output.clone().detach()
            #Reshape since encoder_output shape is (batch,400,3)
            encoder_output = encoder_output.view(-1,20,20,3).cuda()

            #add extra dim for decoder input
            ones = torch.ones((encoder_output.size(0),encoder_output.size(1),encoder_output.size(2),1), requires_grad=True).cuda()
            control_points = torch.cat((encoder_output,ones),-1)

            # print('Decoder input')
            # print(control_points.shape)
            ground_truth_control_points = ground_truth_control_points.view(-1,20,20,3)
            ground_truth_control_points = torch.cat((ground_truth_control_points,ones),-1)
            decoder_output = decoder(control_points)
            ground_truth_decoder = decoder(ground_truth_control_points)
            # print('Decoder output')
            # print(decoder_output.shape)

            decoder_output = decoder_output.view(-1,1600,3)
            ground_truth_decoder = ground_truth_decoder.view(-1,1600,3)

            #compute loss
            decoder_loss,_ = chamfer_distance(decoder_output, ground_truth_decoder)
            print('Decoder Loss', decoder_loss)

            # ===================backward====================
            loss = encoder_loss + decoder_loss
            loss.backward()
            optimizer.step()

            print('Loss:',loss)
      