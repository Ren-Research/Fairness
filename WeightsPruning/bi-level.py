#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, get_user_typep, get_local_wmasks, get_local_bmasks, partition_data
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg2, aggregate, aggregate2
from models.test import test_img


def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm
    keys = list(grad_list.keys())
    
    client_grads = grad_list[keys[0]].view(-1).detach().cpu().numpy()#.view(-1) # shape now: (784, 26)
    #client_grads = np.append(client_grads, grad_list[keys[2]].view(-1)) # output a flattened array
    #print(client_grads)
    for k in keys[1:]:
        client_grads = np.append(client_grads, grad_list[k].view(-1).detach().cpu().numpy()) # output a flattened array
        
        #    for i in range(1, len(grad_list)):
        #        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array--q 1 --device cpu --data_name MNIST --model_name conv --control_name 1_100_0.05_iid_fix_a2-b2-c2-d2-e2_bn_1_1
        #print("grad_list", grad_list, "norm", np.sum(np.square(client_grads)))
    return np.sum(np.square(client_grads))



def get_sub_paras(w_glob, wmask, bmask):
    w_l = copy.deepcopy(w_glob)
    w_l['layer_input.weight'] = w_l['layer_input.weight'] * wmask
    # w_l['layer_input.bias'] = w_l['layer_input.bias'] * bmask

    return w_l


def get_half_paras(w_glob, part_dim):
    w_l = copy.deepcopy(w_glob)
    #print(w_glob.keys(), [w_glob[key].shape for key in w_glob.keys()])
    w_l['layer_input.weight'] = w_l['layer_input.weight'][:part_dim, :]
    w_l['layer_input.bias'] = w_l['layer_input.bias'][:part_dim]
    w_l['layer_hidden.weight'] = w_l['layer_hidden.weight'][:, :part_dim]
    
    #print(w_l.keys(), [w_l[key].shape for key in w_l.keys()])
    
    return w_l


"""
AvgAll:
leave bias alone,
use FedAvg2
"""

if __name__ == '__main__':
    part_dim = 25
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {}
    results["client_train_loss"] = []
    results["client_train_accuracy"] = []
    results["client_test_loss"] = []
    results["client_test_accuracy"] = []
    results["global_model_accuracy"] = []
    results["global_model_loss"] = []
    results["user_idx"] = []
    results["dict_users"] = {}
    results["traindata_cls_counts"] = {}
    results["label_split"] = []
    results["args"] = None
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    results["args"] = vars(args)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        
        # sample users
#        if args.iid:
#            print("===============IID=DATA=======================")
#            dict_users = mnist_iid(dataset_train, args.num_users)
#        else:
#            print("===========NON=IID=DATA=======================")
#            dict_users = mnist_noniid(dataset_train, args.num_users)
#            args.epochs = 100
#            print([len(dict_users[k]) for k in dict_users.keys()])
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
        
    dict_users, traindata_cls_counts = partition_data(dataset_train, args.num_users, args.partition)
    print("dataset", len(dict_users.keys()), [len(dict_users[k]) for k in dict_users.keys()])
    results["dict_users"] = dict_users
    results["traindata_cls_counts"] = traindata_cls_counts
    
    #print("traindata_cls_counts", traindata_cls_counts)
    label_split = [[] for _ in range(len(list(dict_users.keys())))]
    for k in list(dict_users.keys()):
        for key in list(traindata_cls_counts[k].keys()):
            if traindata_cls_counts[k][key] != 0:
                label_split[k].append(key)
    print("label_split", len(label_split), label_split[:10])
    results["label_split"] = label_split
    
    img_size = dataset_train[0][0].shape
    

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        print(net_glob)
        net_part = MLP(dim_in=len_in, dim_hidden=part_dim, dim_out=args.num_classes).to(args.device)
        print(net_part)

    else:
        exit('Error: unrecognized model')
    print(net_glob)

    net_glob.train()
    net_part.train()

    
    w_glob = net_glob.state_dict()
    starting_weights = copy.deepcopy(w_glob)
    
    w_part = get_half_paras(w_glob, part_dim)
    net_part.load_state_dict(w_part)

    setting_arrays = [
        #[1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    setting_array = setting_arrays[0]
    for setting_array in setting_arrays:
        net_glob.load_state_dict(starting_weights)
        # training
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []

        # if args.all_clients:
        #     print("Aggregation over all clients")
        #     w_locals = [w_glob for i in range(args.num_users)]
        print(setting_array)
        setting = str(setting_array).replace(",", "").replace(" ", "").replace("[", "").replace("]", "")
        pic_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
        txt_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.txt'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
        npy_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.npy'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
##########################TEMP USE FOR STRATING LOSS
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        iter = -1
        with open(txt_name, 'a+') as f:
            print('TRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test), file=f)
            print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test), file=f)
        print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
        print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
        net_glob.train()
        ##########################TEMP USE FOR STRATING LOSS

        for iter in range(args.epochs):
            all_clients_epoch_train_loss = []
            all_clients_epoch_train_accuracy = []
            all_clients_epoch_test_loss = []
            all_clients_epoch_test_accuracy = []
            
            if iter > 0:  # >=5 , %5, % 50, ==5
                w_glob = net_glob.state_dict()
                
                w_part = get_half_paras(w_glob, part_dim)
                net_part.load_state_dict(w_part)

    
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            
            np.random.seed(seed)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print(idxs_users)
            results["user_idx"] = idxs_users

            type_array = []
            hs = []
            Deltas = []

            for id, idx in enumerate(idxs_users):
                # typep = get_user_typep(idx, setting_array)
                typep = setting_array[id]
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                if typep == 1:
                    type_array.append(1)
                    w, loss, acc_train, model = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    model.eval()
                    acc_test, loss_test = test_img(model, dataset_test, args)
                    print("# of data samples: " + str(len(dict_users[idx])))
                    print("Full user: " + str(id) + ", training accuracy: " + str(acc_train) + ", training loss: " + str(loss) + ", test accuracy: " + str(acc_test.item()) + ", test loss: " + str(loss_test))
                elif typep == 2:
                    type_array.append(2)
                    #w, loss = local.train(net=copy.deepcopy(net_2).to(args.device))
                    #w = get_sub_paras(w, local_w_masks[1], local_b_masks[1])
                    w, loss, acc_train, model = local.train(net=copy.deepcopy(net_part).to(args.device))
                    model.eval()
                    acc_test, loss_test = test_img(model, dataset_test, args)
                    w = get_half_paras(w, part_dim)
                    print("# of data samples: " + str(len(dict_users[idx])))
                    print("Part user: " + str(id) + ", training accuracy: " + str(acc_train) + ", training loss: " + str(loss) + ", test accuracy: " + str(acc_test.item()) + ", test loss: " + str(loss_test))
                
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                
                # record results
                all_clients_epoch_train_loss.append(loss)
                all_clients_epoch_train_accuracy.append(acc_train)
                all_clients_epoch_test_loss.append(loss_test)
                all_clients_epoch_test_accuracy.append(acc_test.item())
                
            
                if args.q > 0:
                    keys = list(w.keys())
                    grads = copy.deepcopy(w)
                    delta = copy.deepcopy(w)
                    
                    for k in keys:
                        shape = w[k].shape
                        if len(shape) == 2:
                            grads[k] = (w_glob[k][:shape[0], :shape[1]] - w[k]) * 1.0 / args.lr
                        else:
                            grads[k] = (w_glob[k][:shape[0]] - w[k]) * 1.0 / args.lr
                        #print("grads", grads)
                        delta[k] = np.float_power(loss+1e-10, args.q) * grads[k]
                    
                    #loss *= 1000
                    loss = loss
                    # estimation of the local Lipchitz constant
                    hs.append((args.q * np.float_power(loss+1e-10, (args.q-1)) * norm_grad(grads) + (1.0/args.lr) * np.float_power(loss+1e-10, args.q)) / 1000)
                    print("loss", loss, "hs", hs)
                    Deltas.append(delta)
                    #print("deltas", Deltas)
                    
            results["client_train_loss"].append(all_clients_epoch_train_loss)
            results["client_train_accuracy"].append(all_clients_epoch_train_accuracy)
            results["client_test_loss"].append(all_clients_epoch_test_loss)
            results["client_test_accuracy"].append(all_clients_epoch_test_accuracy)
                
    
            # with open(txt_name, 'a+') as f:
            #     print(type_array, file=f)
            print(type_array)
            # FOR ITER

            # update global weights
            #w_glob = FedAvg2(w_locals, type_array, local_w_masks, local_b_masks)
            #w_glob = aggregate(w_glob, 100, w_locals)
            if args.q > 0:
                w_glob = aggregate2(w_glob, hs, Deltas, part_dim, args.device)
            else:
                w_glob = aggregate(w_glob, part_dim, w_locals, args.device)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # with open(txt_name, 'a+') as f:
            #     print(loss_locals, file=f)
            #print(loss_locals)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            with open(txt_name, 'a+') as f:
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg), file=f)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            with open(txt_name, 'a+') as f:
                print('TRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test), file=f)
                print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test), file=f)
            print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
            print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
            net_glob.train()
            
            results["global_model_accuracy"].append(acc_test.item())
            results["global_model_loss"].append(loss_test)
            pickle.dump(results, open("./data/q" + str(args.q) + "_results.pickle", "wb"))  # save it into a file named save.p
            

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