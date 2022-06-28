# -*- coding: utf-8 -*- 
# Author: Jacky
# Creation Date: 2021/3/24

import numpy as np
import torch
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from data_set import convert
from model import loss_func
from torch.autograd import Variable
import copy

class MovieDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
        这里假设，输入的数据集为sample 的矩阵形式
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        # self.user_movie = convert(self.dataset[self.idxs], max(self.dataset[:, 1]))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        ratings = self.dataset[self.idxs[item], 1:-3]
        return torch.tensor(ratings)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # 将输入的dataset转换为user_movie
        self.dataset = convert(dataset[idxs], int(max(dataset[:, 1])))
        self.idxs = np.arange(0, len(self.dataset))
        self.trainloader = DataLoader(MovieDataset(self.dataset, self.idxs),
                                      batch_size=self.args.local_bs, shuffle=True)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to MSE Loss function
        self.criterion = nn.MSELoss().to(self.device)

    def update_weights(self, model, client_idx, global_round):
        """
        训练本地模型，得到模型参数和训练loss
        :param model:
        :param client_idx: 客户0~9
        :param global_round: 全局回合数
        :return: model.state_dict() 模型参数
        :return: sum(epoch_loss) / len(epoch_loss) 本地训练损失
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, betas = (0.9, 0.999), eps = 1e-08,
                                         weight_decay=1e-4)
        # 本地训练 训练回合数设置为local_ep
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, ratings in enumerate(self.trainloader):
                ratings = ratings.to(self.device)

                model.zero_grad()
                outputs = model(ratings)
                loss = self.criterion(outputs, ratings)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 5 == 0):
                    print(
                        '| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, client_idx, iter + 1, batch_idx * len(ratings),
                            len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
                # print(model.state_dict()['linear.weight'].numpy())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss),copy.deepcopy(model)

class Asy_LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # 将输入的dataset转换为user_movie
        self.dataset = convert(dataset[idxs], int(max(dataset[:, 1])))
        self.idxs = np.arange(0, len(self.dataset))
        self.trainloader = DataLoader(MovieDataset(self.dataset, self.idxs),
                                      batch_size=self.args.local_bs, shuffle=True)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to MSE Loss function
        self.criterion = nn.MSELoss().to(self.device)

    def update_weights(self, model, client_idx, global_round,local_learning_rate):
        """
        训练本地模型，得到模型参数和训练loss
        :param model:
        :param client_idx: 客户0~9
        :param global_round: 全局回合数
        :return: model.state_dict() 模型参数
        :return: sum(epoch_loss) / len(epoch_loss) 本地训练损失
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=local_learning_rate,betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=1e-4)
        # 本地训练 训练回合数设置为local_ep
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, ratings in enumerate(self.trainloader):
                ratings = ratings.to(self.device)
                model.zero_grad()
                outputs = model(ratings)
                loss = self.criterion(outputs, ratings)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 5 == 0):
                    print(
                        '| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, client_idx, iter + 1, batch_idx * len(ratings),
                            len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
                # print(model.state_dict()['linear.weight'].numpy())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #print(epoch_loss)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(model)


def cache_hit_ratio(test_dataset, cache_items, request_num):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :return: 缓存命中率
    """
    requset_items = test_dataset[:request_num, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        CACHE_HIT_NUM += count[item]
    CACHE_HIT_RATIO = CACHE_HIT_NUM / len(requset_items) * 100

    return CACHE_HIT_RATIO

def cache_hit_ratio2(test_dataset, cache_items,cache_items2,request_num):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :return: 缓存命中率
    """
    requset_items = test_dataset[:request_num, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        if item not in cache_items2:
            CACHE_HIT_NUM += count[item]
    CACHE_HIT_RATIO = CACHE_HIT_NUM / len(requset_items) * 100

    return CACHE_HIT_RATIO