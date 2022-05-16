#!/usr/bin/env python3

with open('./output/2.txt',"r") as f:
	lines = f.readlines()
	
all_epoch = 0
for line in lines:
	if "Train Epoch:" in line:
		all_epoch = max(all_epoch, int(line.split("Train Epoch:")[1].split("(")[0]))
print(all_epoch)



for i in range(1, all_epoch+1):
	loss = {}
	accuracy = {}
	for line in lines:
		if "Train Epoch: " + str(i) + "(" in line:
			l = float(line.split("Local-Loss:")[1].split("Local-Accuracy:")[0])
			a = float(line.split("Local-Loss:")[1].split("Local-Accuracy:")[1].split("ID:")[0])
			rate = float(line.split("Local-Loss:")[1].split("Local-Accuracy:")[1].split(" Rate:")[1].split(" Epoch")[0])
			#print(l, a, rate)
			if rate not in loss:
				loss[rate] = [l]
			else:
				loss[rate].append(l)
				
			if rate not in accuracy:
				accuracy[rate] = [a]
			else:
				accuracy[rate].append(a)
	#print(loss)
	for key, value in loss.items():
		loss[key] = sum(value) / len(value)
	for key, value in accuracy.items():
		accuracy[key] = sum(value) / len(value)
	print([(k, loss[k]) for k in sorted(list(loss.keys()))])
	print([(k, accuracy[k]) for k in sorted(list(accuracy.keys()))])
	