import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from itertools import chain
import matplotlib.pyplot as plt
from scipy import stats

from options import args_parser
from dataset_processing import sampling, average_weights,asy_average_weights, sampling_mobility
from user_cluster_recommend import recommend, Oracle_recommend
from local_update import LocalUpdate, cache_hit_ratio
from model import AutoEncoder
from utils import exp_details, ModelManager, count_top_items
from data_set import convert
from select_vehicle import select_vehicle, vehicle_p_v, select_vehicle_mobility, vehicle_p_v_mobility, vehicle_p_v_leaving


if __name__ == '__main__':
    idx=0
    # 开始时间
    start_time = time.time()
    # args & 输出实验参数
    args = args_parser()
    exp_details(args)
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load sample users_group_train users_group_test
    sample, users_group_train, users_group_test, request_content, vehicle_request_num = sampling_mobility(args, args.clients_num)
    print('different epoch vehicle request num',vehicle_request_num)

    data_set = np.array(sample)

    # test_dataset & test_dataset_idx
    test_dataset_idxs = []
    for i in range(args.clients_num):
        test_dataset_idxs.append(users_group_test[i])
    test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    test_dataset = data_set[test_dataset_idxs]

    request_dataset = []
    for i in range(args.epochs):
        request_dataset_idxs=[]
        request_dataset_idxs.append(request_content[i])
        request_dataset_idxs = list(chain.from_iterable(request_dataset_idxs))
        request_dataset.append(data_set[request_dataset_idxs])

    all_pos_weight, veh_speed, veh_dis = select_vehicle_mobility(args.clients_num)

    # build model
    global_model = AutoEncoder(int(max(data_set[:, 1])), 100)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    vehicle_model_dict = [[], [], [], [], [], [], [], [], [], []]
    for i in range(args.clients_num):
        vehicle_model_dict[i].append(copy.deepcopy(global_model))
    # copy weights
    global_weights = global_model.state_dict()

    # all epoch weights
    w_all_epochs = dict([(k, []) for k in range(args.epochs)])

    # Training loss
    train_loss = []

    # each epoch train time
    each_epoch_time=[]
    each_epoch_time.append(0)

    cache_efficiency_list=[]
    cache_efficiency_without_list=[]

    while idx < args.epochs:

        # 开始
        print(f'\n | Global Training Round : {idx + 1} |\n')

        global_model.train()

        local_net = copy.deepcopy(vehicle_model_dict[idx % args.clients_num][-1])
        local_net.to(device)

        print("vehicle ", idx % args.clients_num + 1, " start training for ", args.local_ep)

        epoch_start_time = time.time()

        local_weights_avg=[]

        for veh in range(10):

            local_model = LocalUpdate(args=args, dataset=data_set,
                                          idxs=users_group_train[idx % args.clients_num])

            w, loss, local_net = local_model.update_weights(
                model=local_net, client_idx=idx % args.clients_num + 1, global_round=idx + 1)

            local_weights_avg.append(copy.deepcopy(w))

        # update global weights
        global_weights_avg = average_weights(local_weights_avg)
        # update global weights
        global_model.load_state_dict(global_weights_avg)

        epoch_time = time.time() - epoch_start_time
        each_epoch_time.append(epoch_time)

        w_all_epochs[idx] = global_weights_avg['linear1.weight'].tolist()
        cache_size=50

        recommend_movies_c500 = []
        for i in range(args.clients_num):
            vehicle_seq = i
            test_dataset_i = data_set[users_group_test[vehicle_seq]]
            user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
            recommend_list = recommend(user_movie_i, test_dataset_i, w_all_epochs[idx])
            recommend_list500 = count_top_items(cache_size, recommend_list)
            recommend_movies_c500.append(list(recommend_list500))

        # AFPCC
        recommend_movies_c500 = count_top_items(cache_size, recommend_movies_c500)
        all_vehicle_request_num = 0
        for v_num in range(10):
            all_vehicle_request_num += vehicle_request_num[idx][v_num]

        cache_efficiency = cache_hit_ratio(request_dataset[idx], recommend_movies_c500,
                                           all_vehicle_request_num)
        cache_efficiency_list.append(cache_efficiency)

        idx += 1
        veh_dis, veh_speed ,all_pos_weight = vehicle_p_v_mobility(veh_dis , epoch_time, args.clients_num, idx, args.clients_num)


        if idx == args.epochs:

            cache_efficiency_list.insert(0, 0)
            print('Cache hit radio',cache_efficiency_list)


        if idx > args.epochs:
            break

