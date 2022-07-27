#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import torch
import copy
import random


def get_user_typep(x, setting_array=[7, 1, 1, 1]):
    a = 10 * setting_array[0]
    b = a + 10 * setting_array[1]
    c = b + 10 * setting_array[2]

    ############XXXX################
    if 0 <= x < a:  # 0 ~ 70
        return 1
    elif a <= x < b:  # 70 ~ 80
        return 2
    elif b <= x < c:  # 80 ~ 90
        return 3
    elif c <= x < 100:  # 90 ~ 100
        return 4
    else:
        return -1

# torch argsort: 0 being smallest, len(arr) -> largest
# torch where (condition, x(true), y(else))
# 0 ~ 39200 ~ 78400 ~ 117600 ~ 156800
#    4      3       2        1
def get_local_wmasks(ranks):
#    local_masks = []
#    mask = copy.deepcopy(ranks) * 0 + 1
#    local_masks.append(mask.view(200, 784))
#    mask0 = copy.deepcopy(ranks) * 0
#    mask1 = copy.deepcopy(ranks) * 0 + 1
#    # p2
#    x = copy.deepcopy(ranks)
#    mask = torch.where(torch.logical_and(x >= 78400, x < 117600), mask0, mask1)
#    local_masks.append(mask.view(200, 784))
#    # p3
#    x = copy.deepcopy(ranks)
#    mask = torch.where(torch.logical_and(x >= 39200, x < 78400), mask0, mask1)
#    local_masks.append(mask.view(200, 784))
#    # p4
#    x = copy.deepcopy(ranks)
#    mask = torch.where(x < 39200, mask0, mask1)
#    local_masks.append(mask.view(200, 784))
#
#    # p51
#    x = copy.deepcopy(ranks)
#    mask = torch.where(torch.logical_and(x >= 39200, x < 117600), mask0, mask1)
#    local_masks.append(mask.view(200, 784))
#
#    # p52
#    x = copy.deepcopy(ranks)
#    mask = torch.where(torch.logical_and(x >= 78400, x < 117600), mask0, mask1)
#    mask = torch.where(torch.logical_and(x >= 0, x < 39200), mask0, mask)
#    local_masks.append(mask.view(200, 784))
#
#    # p53
#    x = copy.deepcopy(ranks)
#    mask = torch.where(x < 78400, mask0, mask1)
#    local_masks.append(mask.view(200, 784))
#
#    return local_masks
    
    local_masks = []
    mask = copy.deepcopy(ranks) * 0 + 1
    local_masks.append(mask.view(200, 3072))
    mask0 = copy.deepcopy(ranks) * 0
    mask1 = copy.deepcopy(ranks) * 0 + 1
    # p2
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 307200, x < 460800), mask0, mask1)
    local_masks.append(mask.view(200, 3072))
    # p3
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 153600, x < 307200), mask0, mask1)
    local_masks.append(mask.view(200, 3072))
    # p4
    x = copy.deepcopy(ranks)
    mask = torch.where(x < 153600, mask0, mask1)
    local_masks.append(mask.view(200, 3072))
    
    # p51
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 153600, x < 460800), mask0, mask1)
    local_masks.append(mask.view(200, 3072))
    
    # p52
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 307200, x < 460800), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 0, x < 153600), mask0, mask)
    local_masks.append(mask.view(200, 3072))
    
    # p53
    x = copy.deepcopy(ranks)
    mask = torch.where(x < 307200, mask0, mask1)
    local_masks.append(mask.view(200, 3072))
    
    return local_masks






def get_local_bmasks(ranks):
    # [0,50][50,100][100,150][150,200]
    local_masks = []
    mask = copy.deepcopy(ranks) * 0 + 1
    local_masks.append(mask)
    mask0 = copy.deepcopy(ranks) * 0
    mask1 = copy.deepcopy(ranks) * 0 + 1
    # p2
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 100, x < 150), mask0, mask1)
    local_masks.append(mask)
    # p3
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 50, x < 100), mask0, mask1)
    local_masks.append(mask)
    # p4
    x = copy.deepcopy(ranks)
    mask = torch.where(x < 50, mask0, mask1)
    local_masks.append(mask)

    # p51
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 50, x < 150), mask0, mask1)
    local_masks.append(mask)
    # p52
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 50), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 100, x < 150), mask0, mask)
    local_masks.append(mask)
    # p53

    x = copy.deepcopy(ranks)
    mask = torch.where(x >= 100, mask0, mask1)
    local_masks.append(mask)

    return local_masks


def get_mat(p_array, idx):
    x = np.ones(200)
    if idx == 1:
        return x
    for i in range(len(p_array)):
        if p_array[i] == idx:
            x[i] = 0
    return x


def get_matrxs(p_array):
    x = []
    for i in range(1, 5):
        x.append(get_mat(p_array, i))
    return x


def get_onehot_matrixs(rank):
    p_array = get_Pmat(rank)  # P[1,2,3,4] = [0,55,105,155]
    x = get_matrxs(p_array)  # [[1,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
    return x


# Problem?
def get_Pmat(rank):
    def get_P(x):
        if x >= 0 and x < 50:
            return 1
        elif x > 49 and x < 100:
            return 2
        elif x > 99 and x < 150:
            return 3
        elif x > 149 and x < 200:
            return 4
        else:
            return -1

    x = np.zeros(200)
    for i in range(len(rank)):
        x[i] = get_P(rank[i])
    return x


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def partition_data(dataset, num_users, partition="noniid-labeldir", beta=0.4, dataset_name="mnist"):
    if dataset_name == "mnist":
        y_train = dataset.train_labels.numpy()
    elif dataset_name == "cifar10":
        y_train = np.array(dataset.targets)
    n_train = y_train.shape[0]
    
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_users)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_users)}
        
        
    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
            
        N = y_train.shape[0]
        #np.random.seed(2020)
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_users)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_users))
                #print("proportions", proportions)
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
                #print("proportions", proportions)
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                #print("proportions", proportions)
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and num_users <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break
                
                
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_users)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,num_users)
                for j in range(num_users):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(num_users):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_users)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(num_users):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1
                        
    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_users)}
        
    elif partition == "mixed":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
            
        N = y_train.shape[0]
        net_dataidx_map = {}
        
        times=[1 for i in range(10)]
        contain=[]
        for i in range(num_users):
            current=[i%K]
            j=1
            while (j<2):
                ind=random.randint(0,K-1)
                if (ind not in current and times[ind]<2):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_users)}
        
        
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*n_train)
            
        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            
            proportions_k = np.random.dirichlet(np.repeat(beta, 2))
            #proportions_k = np.ndarray(0,dtype=np.float64)
            #for j in range(num_users):
            #    if i in contain[j]:
            #        proportions_k=np.append(proportions_k ,proportions[j])
            
            proportions_k = (np.cumsum(proportions_k)*len(idx_k)).astype(int)[:-1]
            
            split = np.split(idx_k, proportions_k)
            ids=0
            for j in range(num_users):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1
                    
    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, num_users)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(num_users)}
        print(net_dataidx_map)
        for i in range(num_users):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))
    
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (net_dataidx_map, traindata_cls_counts)


def record_net_data_stats(y_train, net_dataidx_map):
    
    net_cls_counts = {}
    
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    
    return net_cls_counts



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    