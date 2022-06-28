import numpy as np
from options import args_parser
args = args_parser()
import scipy.stats as stats

def select_vehicle():
    #each vehicle distance
    #the coverage of RSU 1000m
    dis_round=1000
    vehicle_dis=np.zeros(args.clients_num)
    #vehicle speed 截断高斯分布
    mu, sigma = 38,1
    lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
    x=stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    veh_speed=x.rvs(args.clients_num)
    for i in range(len(veh_speed)):
        veh_speed[i] = veh_speed[i] * 0.278
    print("each vehicle's speed:",veh_speed)
    # veh_age=np.true_divide(dis_round-veh_dis,veh_speed)
    # print("each vehicle's remaining time:",veh_age)
    all_pos_weight=[]
    #sel_turn={}
    for i in range(args.clients_num):
        all_pos_weight.append(dis_round/dis_round)
    return all_pos_weight, veh_speed, vehicle_dis

def select_vehicle_mobility(client_num):
    #each vehicle distance
    #the coverage of RSU 1000m
    dis_round=1000
    vehicle_dis=np.zeros(client_num)
    #vehicle speed 截断高斯分布
    mu, sigma = 38,1
    lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
    x=stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    veh_speed=x.rvs(client_num)
    for i in range(len(veh_speed)):
        veh_speed[i] = veh_speed[i] * 0.278
    print("each vehicle's speed:",veh_speed,'m/s')
    # veh_age=np.true_divide(dis_round-veh_dis,veh_speed)
    # print("each vehicle's remaining time:",veh_age)
    all_pos_weight=[]
    #sel_turn={}
    for i in range(client_num):
        all_pos_weight.append(dis_round/dis_round)
    return all_pos_weight, veh_speed, vehicle_dis


def vehicle_p_v_mobility(veh_dis,epoch_time,vehicle_number,idx,all_vehicle_num):
    mu, sigma = 38, 1
    lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
    x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    veh_speed = x.rvs(vehicle_number)
    for i in range(len(veh_speed)):
        veh_speed[i] = veh_speed[i] * 0.278
    all_pos_weight = []
    for i in range(len(veh_dis)):
        veh_dis[i]+=veh_speed[i]*epoch_time
        if 0 < veh_dis[i] < 1000:
            if len(all_pos_weight) < all_vehicle_num:
                all_pos_weight.append((1000-veh_dis[i]) / 1000)
            else:
                all_pos_weight[idx%all_vehicle_num] = veh_dis[i] / 1000
        if 1000 < veh_dis[i] < 2000:
            if len(all_pos_weight) < all_vehicle_num:
                all_pos_weight.append((2000-veh_dis[i]) / 1000)
            else:
                all_pos_weight[idx % all_vehicle_num] = (veh_dis[i] - 1000) / 1000
        if 2000 < veh_dis[i] < 3000:
            if len(all_pos_weight) < all_vehicle_num:
                all_pos_weight.append((3000-veh_dis[i]) / 1000)
            else:
                all_pos_weight[idx % all_vehicle_num] = (veh_dis[i] - 2000) / 1000

    if len(veh_dis)<vehicle_number:
        for j in range(vehicle_number-len(veh_dis)):
            veh_dis=np.append(veh_dis,0)
            all_pos_weight.append(0)

    return veh_dis,veh_speed,all_pos_weight

def vehicle_p_v_leaving(veh_dis,epoch_time):
    mu, sigma = 25,2.5
    lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
    x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    veh_speed = x.rvs(len(veh_dis))
    all_pos_weight = []
    for i in range(len(veh_dis)):
        veh_dis[i]+=veh_speed[i]*epoch_time/3
        if 0 < veh_dis[i] < 1000:
            all_pos_weight.append((1000-veh_dis[i]) / 1000)
        if 1000 < veh_dis[i] < 2000:
            all_pos_weight.append((2000-veh_dis[i]) / 1000)
        if 2000 < veh_dis[i] < 3000:
            all_pos_weight.append((3000-veh_dis[i]) / 1000)

    return veh_dis,veh_speed,all_pos_weight


def select_vehicle_density(clients_num):

    #each vehicle distance
    dis_round=4
    veh_dis=4*(np.random.rand(clients_num))
    print('position of the coverage area of each vehicles:',veh_dis)

    #vehicle speed 截断高斯分布
    mu, sigma = 110, 5.8
    lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
    x=stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    veh_speed=x.rvs(clients_num)
    print("each vehicle's speed:",veh_speed)

    veh_age=np.true_divide(dis_round-veh_dis,veh_speed)
    print("each vehicle's remaining time:",veh_age)


    turn=[]
    turn_name=[]
    pos_weight = []
    #sel_turn={}
    for i in range(clients_num):
        if veh_age[i]>=0.006:
            turn.append(i)
            turn_name.append(i+1)
            pos_weight.append(veh_dis[i]/dis_round)
    print('Number of vehicles that meet the conditions:',len(turn))
    print('Selected vehicles:',turn_name)

    return turn,pos_weight