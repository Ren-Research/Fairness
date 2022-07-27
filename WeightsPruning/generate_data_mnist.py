#!/usr/bin/env python3

import numpy as np
import copy
import pickle
from torchvision import datasets, transforms
import torch

from utils.sampling import partition_data


####################################################

num_group_users = [10, 10, 10]
dataset_name = "mnist"
distribution = ["noniid-#label10", "iid-diff-quantity-beta0.8", "iid-diff-quantity-beta1", "homo"]
distribution_test = ["noniid-#label10", "iid-diff-quantity-beta0.8", "iid-diff-quantity-beta1", "homo"]
num_train = 1000
num_test = 1000

####################################################

def partition_with_distribution(distribution, num_group_users, dataset, desire=3000):
	results = {}
	all_dict_users = {}
	all_traindata_cls_counts = {}
	
	idx = 0
	for i in range(len(num_group_users)):
		for j in range(num_group_users[i]):
			dist = distribution[j % len(distribution)]
			print(dist)
			if "beta" in dist:
				beta = float(dist.split("beta")[-1])
				dist = dist.split("beta")[0][:-1]
			else:
				beta = 0.4
			
			torch.manual_seed(idx)
			np.random.seed(idx)
			
			dict_users, traindata_cls_counts = partition_data(dataset, int(len(dataset.data) // desire), dist, beta)
			#print([len(dict_users[k]) for k in dict_users.keys()])
			
			min_diff = float("inf")
			min_idx = 0
			for k in dict_users.keys():
				if abs(len(dict_users[k]) - desire) < min_diff:
					min_idx = k
					min_diff = abs(len(dict_users[k]) - desire)
					
			all_dict_users[idx] = dict_users[min_idx]
			all_traindata_cls_counts[idx] = traindata_cls_counts[min_idx]
			
			idx += 1
					
	print([len(all_dict_users[k]) for k in all_dict_users.keys()], [all_dict_users[k][:10] for k in all_dict_users.keys()], all_traindata_cls_counts)
	
	for m in range(len(all_dict_users)):
		label_split = [[] for _ in range(len(list(all_dict_users.keys())))]
		for k in list(all_dict_users.keys()):
			for key in list(all_traindata_cls_counts[k].keys()):
				if all_traindata_cls_counts[k][key] != 0:
					label_split[k].append(key)
	print("label_split", len(label_split), label_split[:10], all_traindata_cls_counts)
	
	return all_dict_users, all_traindata_cls_counts, label_split


trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

n_samples = len(dataset_train.data)
dataset = copy.deepcopy(dataset_train)
dataset.data = dataset.data[:n_samples]
dataset.targets = dataset.targets[:n_samples]

all_dict_users, all_traindata_cls_counts, label_split = partition_with_distribution(distribution, num_group_users, dataset, num_train)
test_all_dict_users, test_all_traindata_cls_counts, test_label_split = partition_with_distribution(distribution_test, num_group_users, dataset_test, num_test)

results = {}
results["all_dict_users"] = all_dict_users
results["all_traindata_cls_counts"] = all_traindata_cls_counts
results["label_split"] = label_split
results["test_all_dict_users"] = test_all_dict_users
results["test_all_traindata_cls_counts"] = test_all_traindata_cls_counts
results["test_label_split"] = test_label_split

results["dataset"] = dataset
results["distribution"] = distribution
results["distribution_test"] = distribution_test
results["num_group_users"] = num_group_users
results["num_train"] = num_train
results["num_test"] = num_test

print("./input/" + dataset_name + "_" + str(sum(num_group_users)) + "_"  + str(num_train) + "_" + str(num_test) + ".pickle")

pickle.dump(results, open("./input/" + dataset_name + "_" + str(sum(num_group_users)) + "_"  + str(num_train) + "_" + str(num_test) + ".pickle", "wb"))  # save it into a file named save.p