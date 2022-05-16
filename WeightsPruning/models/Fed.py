#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np


def FedAvg(w,type_array,local_w_masks,local_b_masks):
    w_avg = copy.deepcopy(w[0])
    all_w_mask, all_b_masks = get_all_masks(type_array,local_w_masks,local_b_masks)

    keys = list(w_avg.keys())
    k = keys[0]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    # w_avg[k] = w_avg[k] / all_w_mask
    w_avg[k] = torch.div(w_avg[k], all_w_mask)

    k = keys[1]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    # w_avg[k] = torch.div(w_avg[k], len(w)) #when pruning weights only
    w_avg[k] = torch.div(w_avg[k] , all_b_masks) # when pruning weights and bias

    for k in keys[2:]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))

    for e in w_avg:

        w_avg[e] = torch.nan_to_num(w_avg[e], nan = 0)
    return w_avg


def get_all_masks(type_array,local_w_masks,local_b_masks):
    all_w_mask = local_w_masks[0] *0
    all_b_masks = local_b_masks[0] * 0
    for e in type_array:
        all_w_mask += local_w_masks[e - 1]
        all_b_masks += local_b_masks[e - 1]
    return all_w_mask, all_b_masks


def FedAvg2(w,type_array,local_w_masks,local_b_masks):
    w_avg = copy.deepcopy(w[0])
    all_w_mask, all_b_masks = get_all_masks(type_array,local_w_masks,local_b_masks)
    #print("all_w_mask", all_w_mask, "all_b_masks", all_b_masks)

    keys = list(w_avg.keys())
    #print("keys", keys, len(w), w)
    k = keys[0]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    # w_avg[k] = w_avg[k] / all_w_mask
    w_avg[k] = torch.div(w_avg[k], all_w_mask)

    k = keys[1]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], len(w)) #when pruning weights only
    # w_avg[k] = torch.div(w_avg[k] , all_b_masks) # when pruning weights and bias

    for k in keys[2:]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def aggregate2(w_glob, hs, Deltas, f): 
    demominator = np.sum(np.asarray(hs))
    num_clients = len(Deltas)
    scaled_deltas = []
    keys = list(Deltas[0].keys())
    
    for client_delta in Deltas:
        #print(demominator, client_delta)
        for k in keys:
            client_delta[k] = client_delta[k] * 1.0 / demominator 
        
        scaled_deltas.append(client_delta)
#        print(client_delta)
#        print(h)
    
    updates = scaled_deltas[0]
    for k in keys:
        updates[k] *= f[0]
    
    for i in range(1, len(Deltas)):
        for k in keys:
            updates[k] += scaled_deltas[i][k] * f[i]
            #print("updates", updates)
    for k in keys:
        w_glob[k] -= updates[k]
    
    return w_glob



def aggregate(w_glob, dims, w): 
    keys = list(w_glob.keys())
    
    w_new = copy.deepcopy(w)
    for i in range(len(dims)):
        if dims[i] != 200:
            zeros = torch.zeros(200-dims[i], 784)
            w_new[i][keys[0]] = torch.vstack((w[i][keys[0]], zeros))
            
            zeros = torch.tensor([0] * (200-dims[i]))
            w_new[i][keys[1]] = torch.hstack((zeros, w[i][keys[1]]))
            
            zeros = torch.zeros(10, 200-dims[i])
            w_new[i][keys[2]] = torch.hstack((w[i][keys[2]], zeros))
            
        #print(w_new[i])
        
    for k in keys:
        w_glob[k] = 1 / len(dims) * w_new[0][k]
        for i in range(1, len(dims)):
            w_glob[k] += 1 / len(dims) * w_new[i][k]
            
    return w_glob



def aggregate(w_glob, dims, w): 
    # [200, 150, 100, 50, 10, 10]
    # [torch.Size([200, 784]), torch.Size([200]), torch.Size([10, 200]), torch.Size([10])]
    # [torch.Size([150, 784]), torch.Size([150]), torch.Size([10, 150]), torch.Size([10])]
    
    keys = list(w_glob.keys())
    
    w_new = copy.deepcopy(w)
    
    for i in range(len(dims)):
        w_glob[keys[0]][:10, :] -= 1 / 6 * w_new[i][keys[0]][:10, :]
        w_glob[keys[1]][:10] -= 1 / 6 * w_new[i][keys[1]][:10]
        w_glob[keys[2]][:, :10] -= 1 / 6 * w_new[i][keys[2]][:, :10]
        w_glob[keys[3]] -= 1 / 6 * w_new[i][keys[3]]
        
    for i in range(4):
        w_glob[keys[0]][10:50, :] -= 1 / 4 * w_new[i][keys[0]][10:50, :]
        w_glob[keys[1]][10:50] -= 1 / 4 * w_new[i][keys[1]][10:50]
        w_glob[keys[2]][:, 10:50] -= 1 / 4 * w_new[i][keys[2]][:, 10:50]
    
    for i in range(3):
        w_glob[keys[0]][50:100, :] -= 1 / 3 * w_new[i][keys[0]][50:100, :]
        w_glob[keys[1]][50:100] -= 1 / 3 * w_new[i][keys[1]][50:100]
        w_glob[keys[2]][:, 50:100] -= 1 / 3 * w_new[i][keys[2]][:, 50:100]
        
    for i in range(2):
        w_glob[keys[0]][100:150, :] -= 1 / 2 * w_new[i][keys[0]][100:150, :]
        w_glob[keys[1]][100:150] -= 1 / 2 * w_new[i][keys[1]][100:150]
        w_glob[keys[2]][:, 100:150] -= 1 / 2 * w_new[i][keys[2]][:, 100:150]
        
    for i in range(1):
        w_glob[keys[0]][150:200, :] -= 1 / 1 * w_new[i][keys[0]][150:200, :]
        w_glob[keys[1]][150:200] -= 1 / 1 * w_new[i][keys[1]][150:200]
        w_glob[keys[2]][:, 150:200] -= 1 / 1 * w_new[i][keys[2]][:, 150:200]
            
    return w_glob


dims = [200, 150, 100, 50, 10, 10]
dim_set = list(set(dims))
dim_set.sort(reverse = True)
print(dim_set)
print(dims.index(dim_set[-1]))