#!/usr/bin/env python3

import pickle
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib
import numpy as np

plt.rcParams["font.family"] = "Linux Biolinum O"
plt.rcParams.update({'font.size': 33,'font.weight':'bold','pdf.fonttype':42})
plt.rcParams["legend.handlelength"]=1.5

files = ["no-bilevel", "global_q_0.0_group_q_0,0,0_group_dim_200,100,25_group_user_10,10,10", "global_q_0.0_group_q_1,1,1_group_dim_200,100,25_group_user_10,10,10", "global_q_1.0_group_q_0,0,0_group_dim_200,100,25_group_user_10,10,10", "global_q_1.0_group_q_1,1,1_group_dim_200,100,25_group_user_10,10,10", "global_q_1.0_group_q_2,2,2_group_dim_200,100,25_group_user_10,10,10", "global_q_1.0_group_q_3,3,3_group_dim_200,100,25_group_user_10,10,10", "global_q_1.0_group_q_4,4,4_group_dim_200,100,25_group_user_10,10,10", "global_q_2.0_group_q_1,1,1_group_dim_200,100,25_group_user_10,10,10", "global_q_2.0_group_q_2,2,2_group_dim_200,100,25_group_user_10,10,10"]

for file in files:
	results = pickle.load(open("./" + file + ".pickle", "rb"))
	print("individual", np.var(results["client_test_accuracy"][-1]))
	gp1 = np.var(results["client_test_accuracy"][-1][:10])
	gp2 = np.var(results["client_test_accuracy"][-1][10:20])
	gp3 = np.var(results["client_test_accuracy"][-1][20:30])
	print("intra", [gp1, gp2, gp3], "inter", np.var([gp1, gp2, gp3]), file)