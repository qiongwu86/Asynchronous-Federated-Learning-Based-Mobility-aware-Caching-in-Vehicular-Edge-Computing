import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from itertools import chain
import matplotlib.pyplot as plt
from scipy import stats

from options import args_parser
from dataset_processing import sampling, average_weights,asy_average_weights, sampling_mobility, sampling_mobility_density
from user_cluster_recommend import recommend, Oracle_recommend
from local_update import LocalUpdate, cache_hit_ratio,Asy_LocalUpdate
from model import AutoEncoder
from utils import exp_details, ModelManager, count_top_items
from data_set import convert
from select_vehicle import select_vehicle, vehicle_p_v, select_vehicle_mobility, vehicle_p_v_mobility


if __name__ == '__main__':
    # 开始时间
    start_time = time.time()
    # args & 输出实验参数
    args = args_parser()
    exp_details(args)
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # ===============================================
    cache_efficiency_list = []
    # ===============================================

    # load sample users_group_train users_group_test
    sample, users_group_train, users_group_test, request_content, vehicle_request_num = sampling_mobility_density(args,25)
    print('different epoch vehicle request num', vehicle_request_num)

    data_set = np.array(sample)

    # test_dataset & test_dataset_idx
    test_dataset_idxs = []
    for i in range(25):
        test_dataset_idxs.append(users_group_test[i])
    test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    test_dataset = data_set[test_dataset_idxs]

    request_dataset_idxs=[]
    for i in range(25):
        request_dataset_idxs.append(users_group_test[i])
    request_dataset_idxs = list(chain.from_iterable(request_dataset_idxs))
    request_dataset=data_set[request_dataset_idxs]

    for i in range(len(args.vehicle_density)):

        vehicle_density=args.vehicle_density[i]
        print('==================vehicle density:',vehicle_density,'============================')


        idx = 0
        all_pos_weight, veh_speed, veh_dis = select_vehicle_mobility(vehicle_density)


        # build model
        global_model = AutoEncoder(int(max(data_set[:, 1])), 100)

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()

        if vehicle_density == 2:
            vehicle_model_dict = [[],[]]
        if vehicle_density == 5:
            vehicle_model_dict = [[], [], [], [], []]
        if vehicle_density == 10:
            vehicle_model_dict = [[], [], [], [], [], [], [], [], [], []]
        if vehicle_density == 15:
            vehicle_model_dict = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        if vehicle_density == 20:
            vehicle_model_dict = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        if vehicle_density == 25:
            vehicle_model_dict = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

        for j in range(vehicle_density):
            vehicle_model_dict[j].append(copy.deepcopy(global_model))

        # copy weights
        global_weights = global_model.state_dict()

        # all epoch weights
        w_all_epochs = dict([(k, []) for k in range(args.epochs)])

        # Training loss
        train_loss = []

        # each epoch train time
        each_epoch_time = []
        each_epoch_time.append(0)

        vehicle_leaving = []


        while idx < args.epochs:

            # 开始
            print(f'\n | Global Training Round : {idx + 1} |\n')

            global_model.train()

            #each vehicle local learning rate
            local_lr = args.lr * max(1,np.log(max(1,idx)))

            local_net = copy.deepcopy(vehicle_model_dict[idx % vehicle_density][-1])
            local_net.to(device)

            print("vehicle ", idx % vehicle_density + 1, " start training for ", args.local_ep,
                  " epochs with learning rate ",local_lr)

            epoch_start_time = time.time()
            local_model = Asy_LocalUpdate(args=args, dataset=data_set,
                                          idxs=users_group_train[idx % vehicle_density])

            w, loss, local_net = local_model.update_weights(
                model=local_net, client_idx=idx % vehicle_density + 1, global_round=idx + 1,
                local_learning_rate=local_lr)

            vehicle_model_dict[idx % vehicle_density].append(local_net)
            v_w=vehicle_model_dict[idx % vehicle_density][-1].state_dict()

            #local weight * (position weight + v2i rate weight)
            for key in v_w.keys():
                v_w[key] = v_w[key] * all_pos_weight[idx % vehicle_density]


            vehicle_model_dict[idx % vehicle_density][-1].load_state_dict(v_w)

            #aggeration

            for name, param in vehicle_model_dict[idx % vehicle_density][-1].named_parameters():
                for name2, param2 in vehicle_model_dict[idx % vehicle_density][-2].named_parameters():
                    if name == name2:
                        param.data.copy_(args.update_decay * param2.data + param.data)

            global_w = asy_average_weights(l=vehicle_model_dict[idx % vehicle_density][-1], g=global_model
                                           , l_old=vehicle_model_dict[idx % vehicle_density][-2],vehicle_all_num=vehicle_density)

            epoch_time = time.time() - epoch_start_time
            each_epoch_time.append(epoch_time)
            global_model.load_state_dict(global_w)

            w_all_epochs[idx] = global_w['linear1.weight'].tolist()

            if idx == args.epochs-1:

                c_s=50
                recommend_movies_c500=[]

                for j in range(vehicle_density):
                    vehicle_seq = j
                    test_dataset_i = data_set[users_group_test[vehicle_seq]]
                    user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
                    recommend_list = recommend(user_movie_i, test_dataset_i, w_all_epochs[idx])
                    recommend_list500 = count_top_items(c_s, recommend_list)
                    recommend_movies_c500.append(list(recommend_list500))

                # AFPCC
                recommend_movies_c500 = count_top_items(c_s, recommend_movies_c500)
                all_vehicle_request_num = 0
                for v_num in range(25):
                    all_vehicle_request_num += vehicle_request_num[v_num]
                cache_efficiency = cache_hit_ratio(request_dataset, recommend_movies_c500,
                                               all_vehicle_request_num)
                cache_efficiency_list.append(cache_efficiency)


            idx += 1
            veh_dis, veh_speed, all_pos_weight = vehicle_p_v_mobility(veh_dis, epoch_time, vehicle_density, idx, vehicle_density)

            if idx > args.epochs:
                break

    print('cache_efficiency_list',cache_efficiency_list)



