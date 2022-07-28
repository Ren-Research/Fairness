#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import collections


def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm
    keys = list(grad_list.keys())
    
    client_grads = grad_list[keys[0]].view(-1).detach().cpu().numpy()#.view(-1) # shape now: (784, 26)
    #client_grads = np.append(client_grads, grad_list[keys[2]].view(-1)) # output a flattened array
    #print(client_grads)
    for k in keys[1:]:
        #print(client_grads.shape, grad_list[k].shape)
        client_grads = np.append(client_grads, grad_list[k].reshape(-1).detach().cpu().numpy()) # output a flattened array
        
        #    for i in range(1, len(grad_list)):
        #        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array--q 1 --device cpu --data_name MNIST --model_name conv --control_name 1_100_0.05_iid_fix_a2-b2-c2-d2-e2_bn_1_1
        #print("grad_list", grad_list, "norm", np.sum(np.square(client_grads)))
    return np.sum(np.square(client_grads))


def clip_paras(w_glob, lo, hi):
    w_l = copy.deepcopy(w_glob)
    #print(w_glob.keys(), [w_glob[key].shape for key in w_glob.keys()])
    #print(w_l['layer_input.weight'].shape)
    w_l['layer_input.weight'] = w_l['layer_input.weight'][lo:hi, :]
    w_l['layer_input.bias'] = w_l['layer_input.bias'][lo:hi]
    w_l['layer_hidden.weight'] = w_l['layer_hidden.weight'][:, lo:hi]
    
    #print(w_l.keys(), [w_l[key].shape for key in w_l.keys()])
    
    return w_l


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


def aggregate_nofair(all_w, device, user_gm, user_pm, user_group_idx, user_dim, w_glob):
    total_group = len(set(user_group_idx))
    
    dim_to_user = collections.defaultdict(list)  # key: dim, value: [user index]
    for i in range(len(user_dim)):
        dim_to_user[user_dim[i]].append(i)
        
    all_dim = sorted(list(dim_to_user.keys()))
    dim_range_to_user = collections.defaultdict(list)  # key: (dim_lo, dim_hi), value: [(user index, group index)]
    for i in range(len(all_dim)):
        if i == 0:
            key = (0, all_dim[0])
        else:
            key = (all_dim[i-1], all_dim[i])
            
        for j in range(len(user_dim)):
            if user_dim[j] >= all_dim[i]:
                dim_range_to_user[key].append((j, user_group_idx[j]))
    print(dim_range_to_user)
    
    w = copy.deepcopy(w_glob)
    for dim_key in dim_range_to_user.keys():
        u0, g0 = dim_range_to_user[dim_key][0]
        tmp = copy.deepcopy(clip_paras(all_w[u0], dim_key[0], dim_key[1])) 
        for k in w.keys():
            tmp[k] *= user_pm[u0] * user_gm[u0]
        
        
        for i in range(1, len(dim_range_to_user[dim_key])):
            u, g = dim_range_to_user[dim_key][i]
            for k in w.keys():
                tmp[k] += copy.deepcopy(clip_paras(all_w[u], dim_key[0], dim_key[1]))[k] * user_pm[u] * user_gm[u]
        
        w['layer_input.weight'][dim_key[0]:dim_key[1], :] = copy.deepcopy(tmp['layer_input.weight'])
        w['layer_input.bias'][dim_key[0]:dim_key[1]] = copy.deepcopy(tmp['layer_input.bias'])
        w['layer_hidden.weight'][:, dim_key[0]:dim_key[1]] = copy.deepcopy(tmp['layer_hidden.weight'])
        w['layer_hidden.bias'] = copy.deepcopy(tmp['layer_hidden.bias'])
        
    return w
            
    
def aggregate_new(all_grad, all_loss, all_qm, user_dim, global_q, inital_glob, L, w_glob, device, user_gm, user_pm, user_group_idx):   
    print("inital_glob", norm_grad(inital_glob))
    
    total_group = len(set(user_group_idx))
    
    dim_to_user = collections.defaultdict(list)  # key: dim, value: [user index]
    for i in range(len(user_dim)):
        dim_to_user[user_dim[i]].append(i)
        
    all_dim = sorted(list(dim_to_user.keys()))
    dim_range_to_user = collections.defaultdict(list)  # key: (dim_lo, dim_hi), value: [(user index, group index)]
    for i in range(len(all_dim)):
        if i == 0:
            key = (0, all_dim[0])
        else:
            key = (all_dim[i-1], all_dim[i])
            
        for j in range(len(user_dim)):
            if user_dim[j] >= all_dim[i]:
                dim_range_to_user[key].append((j, user_group_idx[j]))
    print(dim_range_to_user)
    
    dim_range_to_delta = collections.defaultdict(list)  # key:  (dim_lo, dim_hi), value: [user delta]
    dim_range_to_hs = collections.defaultdict(list)  # key:  (dim_lo, dim_hi), value: [user hs]
    
    for dim_key in list(dim_range_to_user.keys()):
        for group in range(total_group):
            cur_user = []
            for i in range(len(dim_range_to_user[dim_key])):
                if dim_range_to_user[dim_key][i][1] == group:
                    cur_user.append(dim_range_to_user[dim_key][i][0])
            
            #print("cur_user", group, dim_key, cur_user)
            if len(cur_user) > 0:
                gm = user_gm[cur_user[0]]     
                f_m0 = all_loss[cur_user[0]]
                p_m0 = user_pm[cur_user[0]]
                qm = all_qm[cur_user[0]]
                weighted_f = p_m0 * f_m0 ** (qm + 1)
                delta =  clip_paras(copy.deepcopy(all_grad[cur_user[0]]), dim_key[0], dim_key[1])
                for key in delta.keys():
                    delta[key] *= p_m0 * (qm + 1) * f_m0 ** qm
                    
                for i in range(1, len(cur_user)):
                    user = cur_user[i]
                    
                    f_mi = all_loss[user]
                    p_mi = user_pm[user]
                    weighted_f += p_mi * f_mi ** (qm + 1)
                    
                    for key in delta.keys():
                        delta[key] += p_mi * (qm + 1) * f_mi ** qm * copy.deepcopy(clip_paras(all_grad[user], dim_key[0], dim_key[1])[key])
                        
                for key in delta.keys():
                    delta[key] *= gm / (qm + 1) * weighted_f ** ((global_q - qm) / (qm + 1))
                    
                dim_range_to_delta[dim_key].append(delta)
                
                f_m0 = all_loss[cur_user[0]]
                p_m0 = user_pm[cur_user[0]]
                weighted_f1 = p_m0 * f_m0 ** (qm + 1)
                weighted_f2 = p_m0 * (qm + 1) * f_m0 ** qm * L
                weighted_norm = p_m0 * (qm + 1) * qm * f_m0 ** (qm - 1) * norm_grad(copy.deepcopy(clip_paras(all_grad[cur_user[0]], dim_key[0], dim_key[1]))) / norm_grad(inital_glob)
                delta =  clip_paras(copy.deepcopy(all_grad[cur_user[0]]), dim_key[0], dim_key[1])
                for key in delta.keys():
                    delta[key] *= p_m0 * (qm + 1) * f_m0 ** qm
                    
                for i in range(1, len(cur_user)):
                    user = cur_user[i]
                    
                    f_mi = all_loss[user]
                    p_mi = user_pm[user]
                    weighted_f1 += p_mi * f_mi ** (qm + 1)
                    for key in delta.keys():
                        delta[key] += p_mi * (qm + 1) * f_mi ** qm * copy.deepcopy(clip_paras(all_grad[user], dim_key[0], dim_key[1])[key])
                        
                    weighted_norm += p_mi * (qm + 1) * qm * f_mi ** (qm - 1) * norm_grad(copy.deepcopy(clip_paras(all_grad[user], dim_key[0], dim_key[1]))) / norm_grad(inital_glob)
                    weighted_f2 += p_mi * (qm + 1) * f_mi ** qm * L
                    
                dim_range_to_hs[dim_key].append(gm / (qm + 1) * (global_q - qm) / (qm + 1) * weighted_f1 ** ((global_q - 2 * qm - 1) / (qm + 1)) * (norm_grad(delta) / norm_grad(inital_glob)) + gm / (qm + 1) * weighted_f1 ** ((global_q - qm) / (qm + 1)) * (weighted_norm + weighted_f2) )
                
    #print([inital_glob[k].shape for k in inital_glob.keys()])
#    print("all_grad", all_grad)
#    print([norm_grad(k) for k in all_grad])
#    print("all_loss", all_loss)
#    print("dim_range_to_delta", dim_range_to_delta)
#    print("dim_range_to_hs", dim_range_to_hs)
#    print("user_gm", user_gm)
#    print("user_pm", user_pm)
#    print("all_qm", all_qm)
#    print(h)
    scaled_delta = copy.deepcopy(dim_range_to_delta)
    for dim_key in scaled_delta.keys():
        denom = sum(dim_range_to_hs[dim_key])
        #print(dim_key, denom)
        for i in range(len(scaled_delta[dim_key])):
            for k in scaled_delta[dim_key][i].keys():
                scaled_delta[dim_key][i][k] /= denom
            
            #print(scaled_delta[dim_key][i]["layer_input.weight"].shape[0])
            if scaled_delta[dim_key][i]["layer_input.weight"].shape[0] != 200:
                zeros1 = torch.zeros(dim_key[0], 784).to(device)
                zeros2 = torch.zeros(200-dim_key[1], 784).to(device)
                scaled_delta[dim_key][i]["layer_input.weight"] = torch.vstack((zeros1, scaled_delta[dim_key][i]["layer_input.weight"]))
                scaled_delta[dim_key][i]["layer_input.weight"] = torch.vstack((scaled_delta[dim_key][i]["layer_input.weight"], zeros2))
                
                zeros1 = torch.tensor([0] * dim_key[0]).to(device)
                zeros2 = torch.tensor([0] * (200 - dim_key[1])).to(device)
                scaled_delta[dim_key][i]["layer_input.bias"] = torch.hstack((zeros1, scaled_delta[dim_key][i]["layer_input.bias"]))
                scaled_delta[dim_key][i]["layer_input.bias"] = torch.hstack((scaled_delta[dim_key][i]["layer_input.bias"], zeros2))
                
                zeros1 = torch.zeros(10, dim_key[0]).to(device)
                zeros2 = torch.zeros(10, 200-dim_key[1]).to(device)
                scaled_delta[dim_key][i]["layer_hidden.weight"] = torch.hstack((zeros1, scaled_delta[dim_key][i]["layer_hidden.weight"]))
                scaled_delta[dim_key][i]["layer_hidden.weight"] = torch.hstack((scaled_delta[dim_key][i]["layer_hidden.weight"], zeros2))
                    
    w = copy.deepcopy(w_glob)
    for key in w.keys():
        for d1 in scaled_delta.values():
            for d2 in d1:
                w[key] -= d2[key]
    
    return w


def aggregate_nofair_group(all_w, device, user_gm, user_pm, user_group_idx, user_dim, w_glob):
    group_to_user = collections.defaultdict(list)  # key: group index, value: [user index]
    for i in range(len(user_group_idx)):
        gp = user_group_idx[i]
        group_to_user[gp].append(i)
        
    w = copy.deepcopy(w_glob)
    
    idx = 0
    for gp in group_to_user.keys():
        u0, g0 = group_to_user[gp][0], gp
        tmp = copy.deepcopy(all_w[u0]) 
        for k in w.keys():
            tmp[k] *= user_pm[u0] * user_gm[u0]
            
        for i in range(1, len(group_to_user[gp])):
            u, g = group_to_user[gp][i], gp
            for k in w.keys():
                tmp[k] += copy.deepcopy(all_w[u])[k] * user_pm[u] * user_gm[u]
                
        if tmp["layer_input.weight"].shape[0] != 200:
            dim = user_dim[u0]
            zeros = torch.zeros(200-dim, 784).to(device)
            tmp["layer_input.weight"] = torch.vstack((tmp["layer_input.weight"], zeros))
            
            zeros = torch.tensor([0] * (200-dim)).to(device)
            tmp["layer_input.bias"] = torch.hstack((tmp["layer_input.bias"], zeros))
            
            zeros = torch.zeros(10, 200-dim).to(device)
            tmp["layer_hidden.weight"] = torch.hstack((tmp["layer_hidden.weight"], zeros))
        
        if idx == 0:
            w = copy.deepcopy(tmp)
        else:
            for k in w.keys():
                w[k] += tmp[k]
                
        idx += 1

    return w


def aggregate_group(all_grad, all_loss, all_qm, user_dim, global_q, inital_glob, L, w_glob, device, user_gm, user_pm, user_group_idx):  
    group_to_user = collections.defaultdict(list)  # key: group index, value: [user index]
    for i in range(len(user_group_idx)):
        gp = user_group_idx[i]
        group_to_user[gp].append(i)
        
    group_to_delta = {}  # key:  group index, value: [user delta]
    group_to_hs = {}  # key:  group index, value: [user hs]
    
    for gp in list(group_to_user.keys()):
        cur_user = group_to_user[gp]
        
        gm = user_gm[cur_user[0]]     
        f_m0 = all_loss[cur_user[0]]
        p_m0 = user_pm[cur_user[0]]
        qm = all_qm[cur_user[0]]
        weighted_f = p_m0 * f_m0 ** (qm + 1)
        #delta =  clip_paras(copy.deepcopy(all_grad[cur_user[0]]), dim_key[0], dim_key[1])
        delta = copy.deepcopy(all_grad[cur_user[0]])
        for key in delta.keys():
            delta[key] *= p_m0 * (qm + 1) * f_m0 ** qm
            
        for i in range(1, len(cur_user)):
            user = cur_user[i]
            
            f_mi = all_loss[user]
            p_mi = user_pm[user]
            weighted_f += p_mi * f_mi ** (qm + 1)
            
            for key in delta.keys():
                #delta[key] += p_mi * (qm + 1) * f_mi ** qm * copy.deepcopy(clip_paras(all_grad[user], dim_key[0], dim_key[1])[key])
                delta[key] += p_mi * (qm + 1) * f_mi ** qm * copy.deepcopy(all_grad[user][key])
                
        for key in delta.keys():
            delta[key] *= gm / (qm + 1) * weighted_f ** ((global_q - qm) / (qm + 1))
            
        group_to_delta[gp] = delta
        
        f_m0 = all_loss[cur_user[0]]
        p_m0 = user_pm[cur_user[0]]
        weighted_f1 = p_m0 * f_m0 ** (qm + 1)
        weighted_f2 = p_m0 * (qm + 1) * f_m0 ** qm * L
        #weighted_norm = p_m0 * (qm + 1) * qm * f_m0 ** (qm - 1) * norm_grad(copy.deepcopy(clip_paras(all_grad[cur_user[0]], dim_key[0], dim_key[1]))) / norm_grad(inital_glob)
        weighted_norm = p_m0 * (qm + 1) * qm * f_m0 ** (qm - 1) * norm_grad(copy.deepcopy(all_grad[cur_user[0]])) / norm_grad(inital_glob)
        #delta =  clip_paras(copy.deepcopy(all_grad[cur_user[0]]), dim_key[0], dim_key[1])
        delta =  copy.deepcopy(all_grad[cur_user[0]])
        for key in delta.keys():
            delta[key] *= p_m0 * (qm + 1) * f_m0 ** qm
            
        for i in range(1, len(cur_user)):
            user = cur_user[i]
            
            f_mi = all_loss[user]
            p_mi = user_pm[user]
            weighted_f1 += p_mi * f_mi ** (qm + 1)
            for key in delta.keys():
                #delta[key] += p_mi * (qm + 1) * f_mi ** qm * copy.deepcopy(clip_paras(all_grad[user], dim_key[0], dim_key[1])[key])
                delta[key] += p_mi * (qm + 1) * f_mi ** qm * copy.deepcopy(all_grad[user][key])
                
            #weighted_norm += p_mi * (qm + 1) * qm * f_mi ** (qm - 1) * norm_grad(copy.deepcopy(clip_paras(all_grad[user], dim_key[0], dim_key[1]))) / norm_grad(inital_glob)
            weighted_norm += p_mi * (qm + 1) * qm * f_mi ** (qm - 1) * norm_grad(copy.deepcopy(all_grad[user])) / norm_grad(inital_glob)
            weighted_f2 += p_mi * (qm + 1) * f_mi ** qm * L
            
        group_to_hs[gp] = gm / (qm + 1) * (global_q - qm) / (qm + 1) * weighted_f1 ** ((global_q - 2 * qm - 1) / (qm + 1)) * (norm_grad(delta) / norm_grad(inital_glob)) + gm / (qm + 1) * weighted_f1 ** ((global_q - qm) / (qm + 1)) * (weighted_norm + weighted_f2) 
    
    
    scaled_delta = copy.deepcopy(group_to_delta)
    denom = sum(group_to_hs.values())
    for gp in scaled_delta.keys():
        for k in scaled_delta[gp].keys():
            scaled_delta[gp][k] /= denom
            
        #print("hhhhhh", scaled_delta[gp]["layer_input.weight"].shape[0])
        if scaled_delta[gp]["layer_input.weight"].shape[0] != 200:
            dim = user_dim[group_to_user[gp][0]]
            zeros = torch.zeros(200-dim, 784).to(device)
            scaled_delta[gp]["layer_input.weight"] = torch.vstack((scaled_delta[gp]["layer_input.weight"], zeros))
            
            zeros = torch.tensor([0] * (200-dim)).to(device)
            scaled_delta[gp]["layer_input.bias"] = torch.hstack((scaled_delta[gp]["layer_input.bias"], zeros))
            
            zeros = torch.zeros(10, 200-dim).to(device)
            scaled_delta[gp]["layer_hidden.weight"] = torch.hstack((scaled_delta[gp]["layer_hidden.weight"], zeros))
    
    w = copy.deepcopy(w_glob)
    for key in w.keys():
        for d1 in scaled_delta.values():
            w[key] -= d1[key]
                
    return w



            



def aggregate2(w_glob, hs, Deltas, dim, device): 
    #print("hhh", w_glob)
    demominator = np.sum(np.asarray(hs))
    num_clients = len(Deltas)
    scaled_deltas = []
    keys = list(Deltas[0].keys())
    
    for client_delta in Deltas:
        #print("hhh", client_delta)
        #print(demominator, client_delta)
        for k in keys:
            client_delta[k] = client_delta[k] * 1.0 / demominator 
        
        scaled_deltas.append(client_delta)
#        print(client_delta)
        #print("hhh", client_delta)
    
    for i in range(len(hs)):
        if scaled_deltas[i]["layer_input.weight"].shape[0] != 200:
            zeros = torch.zeros(200-dim, 784).to(device)
            scaled_deltas[i]["layer_input.weight"] = torch.vstack((scaled_deltas[i]["layer_input.weight"], zeros))
            
            zeros = torch.tensor([0] * (200-dim)).to(device)
            scaled_deltas[i]["layer_input.bias"] = torch.hstack((scaled_deltas[i]["layer_input.bias"], zeros))
            
            zeros = torch.zeros(10, 200-dim).to(device)
            scaled_deltas[i]["layer_hidden.weight"] = torch.hstack((scaled_deltas[i]["layer_hidden.weight"], zeros))
    
    updates = scaled_deltas[0]
    #print(w_glob, scaled_deltas)
    
    for i in range(1, len(Deltas)):
        for k in keys:
            updates[k] += scaled_deltas[i][k]
            #print("updates", updates)
    for k in keys:
        w_glob[k] -= updates[k]
    #print(updates)
    
    
    return w_glob



def aggregate(w_glob, device, w_locals, user_dim): 
    w = copy.deepcopy(w_glob)
    
    for i in range(len(w_locals)):
        if w_locals[i]["layer_input.weight"].shape[0] != 200:
            dim = w_locals[i]["layer_input.weight"].shape[0]
            zeros = torch.zeros(200-dim, 784).to(device)
            w_locals[i]["layer_input.weight"] = torch.vstack((w_locals[i]["layer_input.weight"], zeros))
            
            zeros = torch.tensor([0] * (200-dim)).to(device)
            w_locals[i]["layer_input.bias"] = torch.hstack((w_locals[i]["layer_input.bias"], zeros))
            
            zeros = torch.zeros(10, 200-dim).to(device)
            w_locals[i]["layer_hidden.weight"] = torch.hstack((w_locals[i]["layer_hidden.weight"], zeros))
        if i == 0:
            w = copy.deepcopy(w_locals[i])
        else:
            for k in w.keys():
                w[k] += w_locals[i][k]
    
    for k in w.keys():
        w[k] *= 1 / len(w_locals)
        
    return w
    
    
    
    
    
    import collections
    
    dim_to_user = collections.defaultdict(list)  # key: dim, value: [user index]
    for i in range(len(user_dim)):
        dim_to_user[user_dim[i]].append(i)
        
    all_dim = sorted(list(dim_to_user.keys()))
    dim_range_to_user = collections.defaultdict(list)  # key: (dim_lo, dim_hi), value: [user index]
    for i in range(len(all_dim)):
        if i == 0:
            key = (0, all_dim[0])
        else:
            key = (all_dim[i-1], all_dim[i])
            
        for j in range(len(user_dim)):
            if user_dim[j] >= all_dim[i]:
                dim_range_to_user[key].append(j)
    print(dim_range_to_user)
    
    w = copy.deepcopy(w_glob)
    cur = 0
    for dim_key in dim_range_to_user.keys():
        denom = len(dim_range_to_user[dim_key])
        
        tmp = copy.deepcopy(clip_paras(w_locals[dim_range_to_user[dim_key][0]], dim_key[0], dim_key[1]))
        for i in range(1, denom):
            for k in tmp.keys():
                tmp[k] += clip_paras(w_locals[dim_range_to_user[dim_key][i]], dim_key[0], dim_key[1])[k]
            
        for k in tmp.keys():
            tmp[k] *= 1 / denom
            
        if tmp["layer_input.weight"].shape[0] != 200:
            dim = tmp["layer_input.weight"].shape[0]
            zeros = torch.zeros(200-dim, 784).to(device)
            tmp["layer_input.weight"] = torch.vstack((tmp["layer_input.weight"], zeros))
            
            zeros = torch.tensor([0] * (200-dim)).to(device)
            tmp["layer_input.bias"] = torch.hstack((tmp["layer_input.bias"], zeros))
            
            zeros = torch.zeros(10, 200-dim).to(device)
            tmp["layer_hidden.weight"] = torch.hstack((tmp["layer_hidden.weight"], zeros))
            
        if cur == 0:
            w = copy.deepcopy(tmp)
        else:
            for k in tmp.keys():
                w[k] += tmp[k]
                
        cur += 1
        
    return w
    
    
    
    
    keys = list(w_glob.keys())
    
    w_new = copy.deepcopy(w)
    for i in range(idx3, idx4):
        zeros = torch.zeros(200-dim, 784).to(device)
        w_new[i]["layer_input.weight"] = torch.vstack((w[i]["layer_input.weight"], zeros))
        
        zeros = torch.tensor([0] * (200-dim)).to(device)
        w_new[i]["layer_input.bias"] = torch.hstack((zeros, w[i]["layer_input.bias"]))
        
        zeros = torch.zeros(10, 200-dim).to(device)
        w_new[i]["layer_hidden.weight"] = torch.hstack((w[i]["layer_hidden.weight"], zeros))
        
        #print(w_new[i].keys(), [w_new[i][key].shape for key in w_new[i].keys()])
    
#    for i in range(len(keys)):
#        if dims[i] != 200:
#            zeros = torch.zeros(200-dims[i], 784)
#            w_new[i][keys[0]] = torch.vstack((w[i][keys[0]], zeros))
#            
#            zeros = torch.tensor([0] * (200-dims[i]))
#            w_new[i][keys[1]] = torch.hstack((zeros, w[i][keys[1]]))
#            
#            zeros = torch.zeros(10, 200-dims[i])
#            w_new[i][keys[2]] = torch.hstack((w[i][keys[2]], zeros))
            
        #print(w_new[i])
        
    w_glob["layer_input.weight"][:dim, :784] = 1 / idx2 * w_new[0]["layer_input.weight"][:dim, :784]
    for i in range(1, idx2):
        w_glob["layer_input.weight"][:dim, :784] += 1 / idx2 * w_new[i]["layer_input.weight"][:dim, :784]
    w_glob["layer_input.weight"][dim:200, :784] = 1 / idx1 * w_new[0]["layer_input.weight"][dim:200, :784]
    for i in range(1, idx1):
        w_glob["layer_input.weight"][dim:200, :784] += 1 / idx1 * w_new[i]["layer_input.weight"][dim:200, :784]
        
    w_glob["layer_input.bias"][:dim] = 1 / idx2 * w_new[0]["layer_input.bias"][:dim]
    for i in range(1, idx2):
        w_glob["layer_input.bias"][:dim] += 1 / idx2 * w_new[i]["layer_input.bias"][:dim]
    w_glob["layer_input.bias"][dim:200] = 1 / idx1 * w_new[0]["layer_input.bias"][dim:200]
    for i in range(1, idx1):
        w_glob["layer_input.bias"][dim:200] += 1 / idx1 * w_new[i]["layer_input.bias"][dim:200]
        
    w_glob["layer_hidden.weight"][:10, :dim] = 1 / idx2 * w_new[0]["layer_hidden.weight"][:10, :dim]
    for i in range(1, idx2):
        w_glob["layer_hidden.weight"][:10, :dim] += 1 / idx2 * w_new[i]["layer_hidden.weight"][:10, :dim]
    w_glob["layer_hidden.weight"][:10, dim:200] = 1 / idx1 * w_new[0]["layer_hidden.weight"][:10, dim:200]
    for i in range(1, idx1):
        w_glob["layer_hidden.weight"][:10, dim:200] += 1 / idx1 * w_new[i]["layer_hidden.weight"][:10, dim:200]
        
        
#    for k in keys:
#        w_glob[k] = 1 / 10 * w_new[0][k]
#        for i in range(1, 5):
#            w_glob[k] += 1 / 10 * w_new[i][k]
#        for i in range(5, 10):
#            w_glob[k] += 1 / 5 * w_new[i][k]
            
    return w_glob


def local_aggregate(w_glob, hs, Deltas, dim, device): 
    #print("hhh", w_glob)
    demominator = np.sum(np.asarray(hs))
    num_clients = len(Deltas)
    scaled_deltas = []
    keys = list(Deltas[0].keys())
    
    for client_delta in Deltas:
        #print("hhh", client_delta)
        #print(demominator, client_delta)
        for k in keys:
            client_delta[k] = client_delta[k] * 1.0 / demominator 
            
        scaled_deltas.append(client_delta)
#        print(client_delta)
        #print("hhh", client_delta)
        
    for i in range(len(hs)):
        if scaled_deltas[i]["layer_input.weight"].shape[0] != 200:
            zeros = torch.zeros(200-dim, 784).to(device)
            scaled_deltas[i]["layer_input.weight"] = torch.vstack((scaled_deltas[i]["layer_input.weight"], zeros))
            
            zeros = torch.tensor([0] * (200-dim)).to(device)
            scaled_deltas[i]["layer_input.bias"] = torch.hstack((scaled_deltas[i]["layer_input.bias"], zeros))
            
            zeros = torch.zeros(10, 200-dim).to(device)
            scaled_deltas[i]["layer_hidden.weight"] = torch.hstack((scaled_deltas[i]["layer_hidden.weight"], zeros))
            
    updates = scaled_deltas[0]
    #print(w_glob, scaled_deltas)
    
    for i in range(1, len(Deltas)):
        for k in keys:
            updates[k] += scaled_deltas[i][k]
            #print("updates", updates)
    
#    for k in keys:
#        updates[k] /= len(Deltas)
    
    return updates



#def aggregate(w_glob, dims, w): 
#    # [200, 150, 100, 50, 10, 10]
#    # [torch.Size([200, 784]), torch.Size([200]), torch.Size([10, 200]), torch.Size([10])]
#    # [torch.Size([150, 784]), torch.Size([150]), torch.Size([10, 150]), torch.Size([10])]
#    
#    keys = list(w_glob.keys())
#    
#    w_new = copy.deepcopy(w)
#    
#    for i in range(len(dims)):
#        w_glob[keys[0]][:10, :] -= 1 / 6 * w_new[i][keys[0]][:10, :]
#        w_glob[keys[1]][:10] -= 1 / 6 * w_new[i][keys[1]][:10]
#        w_glob[keys[2]][:, :10] -= 1 / 6 * w_new[i][keys[2]][:, :10]
#        w_glob[keys[3]] -= 1 / 6 * w_new[i][keys[3]]
#        
#    for i in range(4):
#        w_glob[keys[0]][10:50, :] -= 1 / 4 * w_new[i][keys[0]][10:50, :]
#        w_glob[keys[1]][10:50] -= 1 / 4 * w_new[i][keys[1]][10:50]
#        w_glob[keys[2]][:, 10:50] -= 1 / 4 * w_new[i][keys[2]][:, 10:50]
#    
#    for i in range(3):
#        w_glob[keys[0]][50:100, :] -= 1 / 3 * w_new[i][keys[0]][50:100, :]
#        w_glob[keys[1]][50:100] -= 1 / 3 * w_new[i][keys[1]][50:100]
#        w_glob[keys[2]][:, 50:100] -= 1 / 3 * w_new[i][keys[2]][:, 50:100]
#        
#    for i in range(2):
#        w_glob[keys[0]][100:150, :] -= 1 / 2 * w_new[i][keys[0]][100:150, :]
#        w_glob[keys[1]][100:150] -= 1 / 2 * w_new[i][keys[1]][100:150]
#        w_glob[keys[2]][:, 100:150] -= 1 / 2 * w_new[i][keys[2]][:, 100:150]
#        
#    for i in range(1):
#        w_glob[keys[0]][150:200, :] -= 1 / 1 * w_new[i][keys[0]][150:200, :]
#        w_glob[keys[1]][150:200] -= 1 / 1 * w_new[i][keys[1]][150:200]
#        w_glob[keys[2]][:, 150:200] -= 1 / 1 * w_new[i][keys[2]][:, 150:200]
#            
#    return w_glob


dims = [200, 150, 100, 50, 10, 10]
dim_set = list(set(dims))
dim_set.sort(reverse = True)
print(dim_set)
print(dims.index(dim_set[-1]))