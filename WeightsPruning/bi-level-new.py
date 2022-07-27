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
from models.Fed import FedAvg, FedAvg2, aggregate_nofair, aggregate_new
from models.test import test_img, test_img_part


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
    dataset_name = "mnist"
    num_train = 1000
    num_test = 1000
    
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
    results["intra_var_acc"] = []
    results["intra_var_loss"] = []
    results["inter_var_acc"] = []
    results["inter_var_loss"] = []
    
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
    
    # divide users into groups
    num_active_users = max(int(args.frac * args.num_users), 1)
    num_group_users = [int(i) for i in args.num_group_users.split(",")] # number of users in each group
    #print(num_group_users, num_active_users)
    assert num_active_users == sum(num_group_users)
    # split active user indexes into groups
    np.random.seed(seed)
    active_user_idx = np.random.choice(range(args.num_users), num_active_users, replace=False)
    #print(active_user_idx)
    results["active_user_idx"] = active_user_idx
    
    group_q = [float(i) for i in args.group_q.split(",")]
    user_q = []
    for i in range(len(num_group_users)):
        user_q.extend([group_q[i]] * num_group_users[i])
    print("user q:", user_q)
    assert len(group_q) == len(num_group_users)
    group_dim = [int(i) for i in args.group_dim.split(",")]
    assert len(group_dim) == len(num_group_users)
    user_dim = []
    for i in range(len(num_group_users)):
        user_dim.extend([group_dim[i]] * num_group_users[i])
    print("user dim:", user_dim)
    
    user_group_idx = []
    for i in range(len(num_group_users)):
        user_group_idx.extend([i] * num_group_users[i])
    print("user group index:", user_group_idx)        
    
    ############################################ sample data ###########################

    n_samples = len(dataset_train.data)
    dataset = copy.deepcopy(dataset_train)
    dataset.data = dataset.data[:n_samples]
    dataset.targets = dataset.targets[:n_samples]
    
    input_data = pickle.load(open("./input/" + dataset_name + "_" + str(sum(num_group_users)) + "_"  + str(num_train) + "_" + str(num_test) + ".pickle", "rb"))
    
    all_dict_users, all_traindata_cls_counts, label_split = input_data["all_dict_users"], input_data["all_traindata_cls_counts"], input_data["label_split"]
    test_all_dict_users, test_all_traindata_cls_counts, test_label_split = input_data["test_all_dict_users"], input_data["test_all_traindata_cls_counts"], input_data["test_label_split"]
    
    all_gm = []
    all_pm = []
    start = 0
    total = sum([len(l) for l in all_dict_users.values()])
    for num in num_group_users:
        cnt = 0
        for i in range(start, start+num):
            cnt += len(all_dict_users[i])
        for i in range(start, start+num):
            all_pm.append(len(all_dict_users[i]) / cnt)
        
        tmp = [cnt / total for _ in range(num)]
        all_gm.extend(tmp)
        
        start += num
    print("all gm: ", all_gm, "all pm:", all_pm)
        
#    print("dataset", len(dict_users.keys()), [len(dict_users[k]) for k in dict_users.keys()])
    results["dict_users"] = all_dict_users
    results["traindata_cls_counts"] = all_traindata_cls_counts
    
    print("label_split", len(label_split), label_split[:10], all_traindata_cls_counts)
    results["label_split"] = label_split
    
    img_size = dataset_train[0][0].shape
    results["test_dict_users"] = test_all_dict_users
    print("test_label_split", len(test_label_split), test_label_split[:10], test_all_traindata_cls_counts)
    results["test_label_split"] = test_label_split
    

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        dim_hidden = 200
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=dim_hidden, dim_out=args.num_classes).to(args.device)
        print(net_glob)
        
        net_part = []
        for part_dim in user_dim:
            net = MLP(dim_in=len_in, dim_hidden=part_dim, dim_out=args.num_classes).to(args.device)
            net_part.append(net)
            print(net)

    else:
        exit('Error: unrecognized model')
    #print(net_glob)

    net_glob.train()
    for net in net_part:
        net.train()

    
    w_glob = net_glob.state_dict()
    starting_weights = copy.deepcopy(w_glob)
    
    for i in range(len(net_part)):
        w_part = get_half_paras(w_glob, user_dim[i])
        net_part[i].load_state_dict(w_part)


    net_glob.load_state_dict(starting_weights)
    # training
    ##########################TEMP USE FOR STRATING LOSS
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    iter = -1
#    with open(txt_name, 'a+') as f:
#        print('TRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test), file=f)
#        print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test), file=f)
    print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
    print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
    net_glob.train()
    ##########################TEMP USE FOR STRATING LOSS

    loss_train = []
    for iter in range(args.epochs):
        all_clients_epoch_train_loss = []
        all_clients_epoch_train_accuracy = []
        all_clients_epoch_test_loss = []
        all_clients_epoch_test_accuracy = []
        
        if iter == 0:
            inital_glob = copy.deepcopy(w_glob)
        
        if iter > 0:  # >=5 , %5, % 50, ==5
            w_glob = net_glob.state_dict()
            
            for i in range(len(net_part)):
                w_part = get_half_paras(w_glob, user_dim[i])
                net_part[i].load_state_dict(w_part)
        
        all_loss = []
        all_grad = []
        loss_locals = []
        w_locals = []
        L = 1 / args.lr
        
        for i in range(len(user_q)):
            model = copy.deepcopy(net_part[i])
            
            if user_dim[i] == dim_hidden:
                print("="*50 + str(i) + " Full User Group " + "="*50)
            else:
                print("="*50 + str(i) + " Part User Group " + "="*50)
                
            model.eval()
            
            #u = active_user_idx[i]
            # local testing global model
            #acc_test, loss_test = test_img_part(model, dict_users_test[u], dataset_test, args)
            acc_test, loss_test = test_img_part(model, test_all_dict_users[i], dataset_test, args)
            print("User: " + str(i) + " # of test data samples: " + str(len(test_all_dict_users[i])) + ", test accuracy: " + str(acc_test.item()) + ", test loss: " + str(loss_test))
            #print("User: " + str(u) + " # of test data samples: " + str(len(dict_users_test[i%5])) + ", test accuracy: " + str(acc_test.item()) + ", test loss: " + str(loss_test))
            
            # local updates
            #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_homo[u])
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=all_dict_users[i])
            w, loss, acc_train, trained_model = local.train(net=copy.deepcopy(model).to(args.device))
            print("# of training data samples: " + str(len(all_dict_users[i])) + ", training accuracy: " + str(acc_train) + ", training loss: " + str(loss))
            #print("# of training data samples: " + str(len(dict_users_homo[i%5])) + ", training accuracy: " + str(acc_train) + ", training loss: " + str(loss))
            print()
            
            #w = get_half_paras(w, user_dim[i])
            
            all_clients_epoch_train_loss.append(loss)
            all_clients_epoch_train_accuracy.append(acc_train)
            all_clients_epoch_test_loss.append(loss_test)
            all_clients_epoch_test_accuracy.append(acc_test)
            
            if args.bilevel and iter > 0:
                if args.test_loss:
                    l = copy.deepcopy(loss_test)
                else:
                    l = copy.deepcopy(loss)
                    
                all_loss.append(l)
                
                update = copy.deepcopy(w)
                keys = update.keys()
                for k in keys:
                    shape = w[k].shape
                    if len(shape) == 2:
                        update[k] = (w_glob[k][:shape[0], :shape[1]] - w[k]) * L
                    else:
                        update[k] = (w_glob[k][:shape[0]] - w[k]) * L
                all_grad.append(update)
                
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
                
        results["client_train_loss"].append(all_clients_epoch_train_loss)
        results["client_train_accuracy"].append(all_clients_epoch_train_accuracy)
        results["client_test_loss"].append(all_clients_epoch_test_loss)
        results["client_test_accuracy"].append(all_clients_epoch_test_accuracy)
        
        # compute variance
        print("="*50 + " Compute Variance " + "="*50)
        inter_acc = []
        inter_loss = []
        intra_var_acc_epoch = []
        intra_var_loss_epoch = []
        for i in range(len(num_group_users)):
            num = num_group_users[i]
            if i == 0:
                var_acc = np.var(all_clients_epoch_test_accuracy[:num])
                var_loss = np.var(all_clients_epoch_test_loss[:num])
                mean_acc_1 = np.mean(all_clients_epoch_test_accuracy[:num])
                mean_loss_1 = np.mean(all_clients_epoch_test_loss[:num])
                inter_acc.append(mean_acc_1)
                inter_loss.append(mean_loss_1)
            else:
                num2 = sum(num_group_users[:i])
                var_acc = np.var(all_clients_epoch_test_accuracy[num2:num+num2])
                var_loss = np.var(all_clients_epoch_test_loss[num2:num+num2])
                mean_acc_1 = np.mean(all_clients_epoch_test_accuracy[num2:num+num2])
                mean_loss_1 = np.mean(all_clients_epoch_test_loss[num2:num+num2])
                inter_acc.append(mean_acc_1)
                inter_loss.append(mean_loss_1)
            intra_var_acc_epoch.append(var_acc)
            intra_var_loss_epoch.append(var_loss)
            print("Intra-group Accuracy variance: " + str(var_acc) + ", Loss variance: " + str(var_loss))
        #print(inter_acc, inter_loss)
        print("Inter-group accuracy variance: " + str(np.var(inter_acc)))
        print("Inter-group loss variance: " + str(np.var(inter_loss)))
        results["intra_var_acc"].append(intra_var_acc_epoch)
        results["intra_var_loss"].append(intra_var_loss_epoch)
        results["inter_var_acc"].append(np.var(inter_acc))
        results["inter_var_loss"].append(np.var(inter_loss))
        
            
        # update global weights
        #w_glob = FedAvg2(w_locals, type_array, local_w_masks, local_b_masks)
        #w_glob = aggregate(w_glob, 100, w_locals)
        if args.bilevel and iter > 0:
            w_glob = aggregate_new(all_grad, all_loss, user_q, user_dim, args.global_q, inital_glob, L, w_glob, args.device, all_gm, all_pm, user_group_idx)
        else:
            #w_glob = aggregate_new(all_grad, all_loss, [0] * len(user_q), user_dim, 0, w_glob, L, w_glob, args.device, all_gm, all_pm, user_group_idx)
            w_glob = aggregate_nofair(w_locals, args.device, all_gm, all_pm, user_group_idx, user_dim, w_glob)
            
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # with open(txt_name, 'a+') as f:
        #     print(loss_locals, file=f)
        #print(loss_locals)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
#        with open(txt_name, 'a+') as f:
#            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg), file=f)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
#        with open(txt_name, 'a+') as f:
#            print('TRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test), file=f)
#            print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test), file=f)
        print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
        print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
        net_glob.train()
        
        results["global_model_accuracy"].append(acc_test.item())
        results["global_model_loss"].append(loss_test)
        
        if not args.bilevel:
            file_name = "no-bilevel"
        else:
            file_name = "global_q_" + str(args.global_q) + "_group_q_" + str(args.group_q) + "_group_dim_" + str(args.group_dim) + "_group_user_" + str(args.num_group_users)
        pickle.dump(results, open("./data/tmp/" + file_name + ".pickle", "wb"))  # save it into a file named save.p
        
        #print(results)

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
#    with open(txt_name, 'a+') as f:
#        print("Training accuracy: {:.2f}".format(acc_train), file=f)
#        print("Testing accuracy: {:.2f}".format(acc_test), file=f)
    # np.save(npy_name, loss_train)
    print("Training accuracy: {:.2f}==================================================".format(acc_train))
    print("Testing accuracy: {:.2f}==================================================".format(acc_test))