#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, get_user_typep, get_local_wmasks, get_local_bmasks
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg2, aggregate2, aggregate
from models.test import test_img


def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm
    keys = list(grad_list.keys())
    
    client_grads = grad_list[keys[0]].view(-1)#.view(-1) # shape now: (784, 26)
    #client_grads = np.append(client_grads, grad_list[keys[2]].view(-1)) # output a flattened array
    #print(client_grads)
    for k in keys[1:]:
        client_grads = np.append(client_grads, grad_list[k].view(-1)) # output a flattened array
    
#    for i in range(1, len(grad_list)):
#        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array
        
    return np.sqrt(np.sum(np.square(client_grads)))


#def get_sub_paras(w_glob, wmask, bmask):
#    w_l = copy.deepcopy(w_glob)
#    w_l['layer_input.weight'] = w_l['layer_input.weight'] * wmask
#    # w_l['layer_input.bias'] = w_l['layer_input.bias'] * bmask
#
#    return w_l


def get_sub_paras(w_glob, dim):
    w_l = copy.deepcopy(w_glob)
    keys = list(w_l.keys())
    
    w_l[keys[0]] = w_glob[keys[0]][:dim, :]
    w_l[keys[1]] = w_glob[keys[1]][:dim]
    w_l[keys[2]] = w_glob[keys[2]][:, :dim]
    
    return w_l


"""
AvgAll:

leave bias alone,
use FedAvg2

"""

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print("===============IID=DATA=======================")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("===========NON==ID=DATA=======================")
            dict_users = mnist_noniid(dataset_train, args.num_users)
            #args.epochs = 100
        #print([len(dict_users[k]) for k in dict_users.keys()])
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    print("image size", img_size)
    print("learning rate", args.lr, "epoch", args.epochs)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        
        net_2 = CNNMnist(args=args).to(args.device)
        net_3 = CNNMnist(args=args).to(args.device)
        net_4 = CNNMnist(args=args).to(args.device)
        
        net_51 = CNNMnist(args=args).to(args.device)
        net_52 = CNNMnist(args=args).to(args.device)
        net_53 = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

        net_2 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_3 = MLP(dim_in=len_in, dim_hidden=150, dim_out=args.num_classes).to(args.device)
        net_4 = MLP(dim_in=len_in, dim_hidden=100, dim_out=args.num_classes).to(args.device)

        net_51 = MLP(dim_in=len_in, dim_hidden=50, dim_out=args.num_classes).to(args.device)
        net_52 = MLP(dim_in=len_in, dim_hidden=10, dim_out=args.num_classes).to(args.device)
        net_53 = MLP(dim_in=len_in, dim_hidden=10, dim_out=args.num_classes).to(args.device)

    else:
        exit('Error: unrecognized model')
    print(net_glob)

    net_glob.train()
    net_2.train()
    net_3.train()
    net_4.train()
    net_51.train()
    net_52.train()
    net_53.train()
    
    print([net_glob.state_dict()[k].shape for k in net_glob.state_dict().keys()])
    print([net_3.state_dict()[k].shape for k in net_glob.state_dict().keys()])
    
    print("***********MODIFIED: PRUNING WEIGHTS ONLY******************")
    # copy weights

    # Ranking the paras
    # w_glob['layer_input.weight'].view(-1).view(200, 784)  #156800 = 39200 * 4
    # Smallest  [0,39200], [39200,78400], [78400,117600],[117600, 156800] (largest)
    """
1 w_net      X              X                X               X
2 net_2      X              X                                X
3 net_3      X                               X               X
4 net_4                     X                X               X

5 net_51     X                                               X
6 net_52                    X                                X
7 net_53                                     X               X
    """
    #    [0,50]         [50,100]         [100,150]        [150,200]
    w_glob = net_glob.state_dict()
    #print(w_glob)
    starting_weights = copy.deepcopy(w_glob)

    w_n2 = get_sub_paras(w_glob, 200)
    net_2.load_state_dict(w_n2)
    w_n3 = get_sub_paras(w_glob, 150)
    net_3.load_state_dict(w_n3)
    w_n4 = get_sub_paras(w_glob, 100)
    net_4.load_state_dict(w_n4)

    w_n51 = get_sub_paras(w_glob, 50)
    net_51.load_state_dict(w_n51)
    w_n52 = get_sub_paras(w_glob, 10)
    net_52.load_state_dict(w_n52)
    w_n53 = get_sub_paras(w_glob, 10)
    net_53.load_state_dict(w_n53)


    net_glob.load_state_dict(starting_weights)
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
        
##########################TEMP USE FOR STRATING LOSS
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    iter = -1
    
    print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
    print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
    net_glob.train()
    ##########################TEMP USE FOR STRATING LOSS

    for iter in range(args.epochs):
        Deltas = []
        hs = []

        if iter > 0:  # >=5 , %5, % 50, ==5
            w_glob = net_glob.state_dict()

            w_n2 = get_sub_paras(w_glob, 200)
            net_2.load_state_dict(w_n2)
            w_n3 = get_sub_paras(w_glob, 150)
            net_3.load_state_dict(w_n3)
            w_n4 = get_sub_paras(w_glob, 100)
            net_4.load_state_dict(w_n4)
        
            w_n51 = get_sub_paras(w_glob, 50)
            net_51.load_state_dict(w_n51)
            w_n52 = get_sub_paras(w_glob, 10)
            net_52.load_state_dict(w_n52)
            w_n53 = get_sub_paras(w_glob, 10)
            net_53.load_state_dict(w_n53)

            """
            net_glob : full net
            net_2~net_4: 75% net 
            net_51 ~ net_53 50% net     
            """
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)

        for id, idx in enumerate(idxs_users):
            # typep = get_user_typep(idx, setting_array)
            
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if id == 0:
                w, loss = local.train(net=copy.deepcopy(net_2).to(args.device))
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
            
            elif id == 1:
                w, loss = local.train(net=copy.deepcopy(net_3).to(args.device))
                acc_test, loss_test = test_img(net_2, dataset_test, args)
            
            elif id == 2:
                w, loss = local.train(net=copy.deepcopy(net_4).to(args.device))
                acc_test, loss_test = test_img(net_3, dataset_test, args)
                
            elif id == 3:
                w, loss = local.train(net=copy.deepcopy(net_51).to(args.device))
                acc_test, loss_test = test_img(net_4, dataset_test, args)

            elif id == 4:
                w, loss = local.train(net=copy.deepcopy(net_52).to(args.device))
                acc_test, loss_test = test_img(net_51, dataset_test, args)
                
            elif id == 5:
                w, loss = local.train(net=copy.deepcopy(net_53).to(args.device))
                acc_test, loss_test = test_img(net_51, dataset_test, args)
                
    
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
            print("client accuracy", id, acc_test, loss)
            # compute for fairness
            #print(w_glob, w)
            
#            f = 1
#            
#            keys = list(w.keys())
#            
#            grads = copy.deepcopy(w)
#            delta = copy.deepcopy(w)
#            
#            for k in keys:
#                grads[k] = (w_glob[k] - w[k]) * 1.0 / args.lr
#                #print("grads", grads)
#                delta[k] = np.float_power(loss+1e-10, args.q) * grads[k] * (f**args.q)
#                #print("delta", delta)
##                    Deltas.append([np.float_power(loss+1e-10, args.q) * grad for grad in grads])
##                    
##                    hs[k] = args.q * np.float_power(loss+1e-10, (args.q-1)) * norm_grad(grads) + (1.0/args.lr) * np.float_power(loss+1e-10, args.q)
#                
#            # estimation of the local Lipchitz constant
#            hs.append(args.q * np.float_power(loss+1e-10, (args.q-1)) * norm_grad(grads)  * (f**args.q) + (1.0/args.lr) * np.float_power(loss+1e-10, args.q)  * (f**args.q))
#            #print("hs", hs)
#            Deltas.append(delta)
#            #print("deltas", Deltas)
                            

        # FOR ITER

        # update global weights [1.59, 1.59, 1.59, 1.59, 1.19, 1.19, 1.19, 1.19, 0.794, 0.794] [0.794**args.q, 0.794**args.q, 0.794**args.q, 0.794**args.q, 1.19**args.q, 1.19**args.q, 1.19**args.q, 1.19**args.q, 1.59**args.q, 1.59**args.q]
        w_glob = aggregate(w_glob, [200, 150, 100, 50, 10, 10], w_locals)
        #print(w_glob)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # with open(txt_name, 'a+') as f:
        #     print(loss_locals, file=f)
        print(loss_locals)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        
        print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
        print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
        net_glob.train()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(pic_name)
    np.save(npy_name, loss_train)
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    with open(txt_name, 'a+') as f:
        print("Training accuracy: {:.2f}".format(acc_train), file=f)
        print("Testing accuracy: {:.2f}".format(acc_test), file=f)
    # np.save(npy_name, loss_train)
    print("Training accuracy: {:.2f}==================================================".format(acc_train))
    print("Testing accuracy: {:.2f}==================================================".format(acc_test))
        