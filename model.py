# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/3/8


# Importing the libraries
import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class AutoEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        """
        AutoEncoder初始化
        """
        super(AutoEncoder, self).__init__()
        # 线性方程和激励函数
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_in)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        神经网络forward函数
        :param x: 输入，Input
        :return: y_pred
        """
        y_encode = self.linear1(x)
        y_pred = self.sigmoid(self.linear2(y_encode))
        return y_pred


class VAE(nn.Module):

  def __init__(self, nb_movies, device="cuda:0"):
    super(VAE, self).__init__()
    self.nb_movies = nb_movies
    self.encoder = nn.Sequential(
        nn.Linear(self.nb_movies, 512),
        nn.Sigmoid(),
        nn.Dropout(0.9), # 需要一个较大的dropout
        nn.Linear(512, 80),
        nn.Sigmoid()
        )
    self.fc1 = nn.Linear(80, 32)
    self.fc2 = nn.Linear(80, 32)
    self.decoder = nn.Sequential(
        nn.Linear(32, 80),
        nn.Sigmoid(),
        nn.Linear(80, 512),
        nn.Sigmoid(),
        nn.Linear(512, self.nb_movies)
        )

  # reparameterize
  def reparameterize(self, mu, logvar):
    eps = Variable(torch.randn(mu.size(0), mu.size(1)))
    z = mu + eps * torch.exp(logvar/2)
    return z

  def forward(self, x):
    out1, out2 = self.encoder(x), self.encoder(x)
    mu = self.fc1(out1)
    logvar = self.fc2(out2)
    z = self.reparameterize(mu, logvar)
    return self.decoder(z), mu, logvar

def loss_func(recon_x, x, mu, logvar):
  """
  VAE的损失包括两部分：
  一部分是预测结果和真实结果的平均绝对误差；
  另一部分是KL-divergence（KL散度），用来衡量潜在变量的分布和单位高斯分布的差异。
  """
  MSE = torch.mean(torch.norm((x - recon_x), p=2, dim=1, keepdim=False)**2/torch.sum(recon_x!=0,axis=1))
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return MSE + KLD


class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals