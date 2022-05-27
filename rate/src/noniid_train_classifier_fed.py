import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed import Federation
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger

from noniid_model import *
from noniid_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
parser.add_argument('--q', default=0, type=float)
parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
parser.add_argument('--beta', type=float, default=0.5, help='The concentration parameter of the Dirichlet distribution for heterogeneous partition')

args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
    if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
    'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}

results = {"user_idx": [], "local_train_loss": [], "local_train_accuracy": [], "local_test_loss": [], "local_test_accuracy": [], "global_test_loss": [], "global_test_accuracy": [], "global_model_loss": [], "global_model_accuracy": []}


def main():
    process_control()
    print("args", args)
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    print("fetch_dataset", dataset)
    
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    print("model", model)
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
            optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    print("data_split", data_split.keys(), "train", [len(data_split["train"][i]) for i in data_split["train"].keys()], "test", [len(data_split["test"][i]) for i in data_split["test"].keys()])
    print("label_split", [len(label_split[i]) for i in label_split.keys()])
    
    seed = cfg['init_seed']
    print("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    print("Partitioning data")
    #    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
    #      'cifar10', './data/{}'.format('cifar10'), './data/{}'.format(cfg['data_name']), args.partition, args.n_parties, beta=args.beta)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        'cifar10', './data/{}'.format('CIFAR10'), './data/{}'.format(cfg['data_name']), args["partition"], cfg['num_users'], beta=args["beta"])
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    print("net_dataidx_map", net_dataidx_map.keys(), [len(net_dataidx_map[i]) for i in list(net_dataidx_map.keys())])
    print("traindata_cls_counts", traindata_cls_counts.keys())
    
    print("changing local training data split, test data split remains the same (iid)")
    data_split['train'] = net_dataidx_map
    
    label_split = [[] for _ in range(len(list(net_dataidx_map.keys())))]
    for k in list(net_dataidx_map.keys()):
        for key in list(traindata_cls_counts[k].keys()):
            if traindata_cls_counts[k][key] != 0:
                label_split[k].append(key)
    print("label_split", len(label_split), label_split[:10])
    
    
    global_parameters = model.state_dict()
    federation = Federation(global_parameters, cfg['model_rate'], label_split)
    
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        print("*"*30, "Global epoch:", epoch, "*"*30)
        if epoch == 1:
            train(dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch, first=True, test_dataset=dataset['test'], test_data_split=data_split['test'])
        else:
            train(dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch, first=False, test_dataset=dataset['test'], test_data_split=data_split['test'])
        
        print("="*25, "Test global model on global test dataset", "="*25)
        test_model = stats(dataset['train'], model)
        loss, accuracy = test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch, final=True)
        results["global_model_loss"].append(loss)
        results["global_model_accuracy"].append(accuracy)
        print(results)
        pickle.dump(results, open("./data/q" + str(args["q"]) + "_f" + str(1) + "_results.pickle", "wb"))  # save it into a file named save.p
        
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm
    keys = list(grad_list.keys())
    
    client_grads = grad_list[keys[0]].view(-1).cpu().numpy()#.view(-1) # shape now: (784, 26)
    #client_grads = np.append(client_grads, grad_list[keys[2]].view(-1)) # output a flattened array
    #print(client_grads)
    for k in keys[1:]:
        client_grads = np.append(client_grads, grad_list[k].view(-1).cpu().numpy()) # output a flattened array
        
        #    for i in range(1, len(grad_list)):
        #        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array--q 1 --device cpu --data_name MNIST --model_name conv --control_name 1_100_0.05_iid_fix_a2-b2-c2-d2-e2_bn_1_1
        #print("grad_list", grad_list, "norm", np.sum(np.square(client_grads)))
    return np.sum(np.square(client_grads))



def train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, first=False, test_dataset=None, test_data_split=None):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    start_time = time.time()
    
    hs = []
    Deltas = []
    
    local_train_loss = []
    local_train_accuracy = []
    local_test_loss = []
    local_test_accuracy = []
    global_test_loss = []
    global_test_accuracy = []
    for m in range(num_active_users):
        print("="*25, "Local client:", m, ", User:", user_idx[m], "="*25)
        print("Training on local training dataset, number of local data samples:", len(data_split[user_idx[m]]), ", local labels:", label_split[user_idx[m]])
        param, (train_loss, train_accuracy), local_model = local[m].train(local_parameters[m], lr, logger)
        local_train_loss.append(train_loss)
        local_train_accuracy.append(train_accuracy)
        local_parameters[m] = copy.deepcopy(param)
        #print(loss, accuracy)
        #local_parameters[m] = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        if True:
            #if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                'Learning rate: {}'.format(lr),
                'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                'Epoch Finished Time: {}'.format(epoch_finished_time),
                'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            info = logger.write('train', cfg['metric_name']['train']['Local'])
            
#            loss = float(info.split("Local-Loss:")[1].split("Local-Accuracy:")[0])
#            accuracy = float(info.split("Local-Accuracy:")[1].split("ID:")[0])
#            print(loss, accuracy)
            
#        if (epoch - 1) % 10 == 0:
#            print("Test on global testing dataset, epoch:", epoch)
#            #test_model = stats(dataset, local_model)
#            test(test_dataset, test_data_split, label_split, local_model, logger, epoch)
#            #test_img(local_model, test_dataset)
            
        print("Test on local test dataset")
        test_loss, test_accuracy = test(test_dataset, test_data_split, label_split, local_model, logger, epoch, local_test=True, local_idx=user_idx[m])
        local_test_loss.append(test_loss)
        local_test_accuracy.append(test_accuracy)
        
        if not first and args["q"] != 0:    
            # for fairness
            loss = train_loss
            loss *= 1000
            f = federation.model_rate[user_idx[m]]
            #f = 1
            loss *= f
            print("loss", loss)
            
            w = copy.deepcopy(local_parameters[m])
            w_glob = copy.deepcopy(federation.global_parameters)
            keys = list(w.keys())
            
            grads = OrderedDict()
            delta = OrderedDict()
            
            for k in keys:
                #print(w_glob[k].shape, w[k].shape)
                dim = w[k].shape
                if len(dim) == 4:
                    grads[k] = (w_glob[k][:dim[0], :dim[1], :dim[2], :dim[3]] - w[k]) * 1.0 / lr
                elif len(dim) == 3:
                    grads[k] = (w_glob[k][:dim[0], :dim[1], :dim[2]] - w[k]) * 1.0 / lr
                elif len(dim) == 2:
                    grads[k] = (w_glob[k][:dim[0], :dim[1]] - w[k]) * 1.0 / lr
                elif len(dim) == 1:
                    grads[k] = (w_glob[k][:dim[0]] - w[k]) * 1.0 / lr
                    #print("grads", grads)
                    #delta[k] = np.float_power(loss+1e-10, args["q"]) * grads[k] * (f**args["q"])
                delta[k] = np.float_power(loss, args["q"]) * grads[k]
                #print("delta", delta)
                
                # estimation of the local Lipchitz constant
                #hs.append(args["q"] * np.float_power(loss+1e-10, (args["q"]-1)) * norm_grad(grads)  * (f**args["q"]) + (1.0/lr) * np.float_power(loss+1e-10, args["q"])  * (f**args["q"]))
            hs.append(args["q"] * np.float_power(loss, (args["q"]-1)) * norm_grad(grads) + (1.0/lr) * np.float_power(loss, args["q"]))
            print("hs", hs)
            Deltas.append(delta)
            #print("deltas", Deltas)
        
        if epoch == cfg['num_epochs']['global']:
            print("For the last global epoch, test on global test dataset")
            global_loss, global_accuracy = test(test_dataset, test_data_split, label_split, local_model, logger, epoch, local_test=False, local_idx=user_idx[m])
            global_test_loss.append(global_loss)
            global_test_accuracy.append(global_accuracy)
            
        if m == num_active_users - 1 and epoch == cfg['num_epochs']['global']:
            results["global_test_loss"].append(global_test_loss)
            results["global_test_accuracy"].append(global_test_accuracy)
            
    if first or args["q"] == 0:
        federation.combine(local_parameters, param_idx, user_idx)
    else:
        federation.combine2(local_parameters, param_idx, user_idx, hs, Deltas)
        
    global_model.load_state_dict(federation.global_parameters)
    
    results["local_train_loss"].append(local_train_loss)
    results["local_train_accuracy"].append(local_train_accuracy)
    results["local_test_loss"].append(local_test_loss)
    results["local_test_accuracy"].append(local_test_accuracy)
    
    print(results)
    
    return


def stats(dataset, model):
    with torch.no_grad():
        test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"])'
            .format(cfg['model_name']))
        test_model.load_state_dict(model.state_dict(), strict=False)
        data_loader = make_data_loader({'train': dataset})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def test(dataset, data_split, label_split, model, logger, epoch, final=False, local_test=False, local_idx=-1):
    #print(model)
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        
        # if testing on local client
        if local_test:
            m = local_idx
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            epoch_loss = []
            epoch_accuracy = []
            #compute_acc = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                output = model(input)
                #print(output)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
                #print(output['loss'].mean(), evaluation["Local-Loss"])
                epoch_loss.append(evaluation["Local-Loss"])
                epoch_accuracy.append(evaluation["Local-Accuracy"])
                #compute_acc.append(compute_accuracy(output["score"], input['label']))
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            mean_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
            #mean_compute_accuracy = sum(compute_acc) / len(compute_acc)
            print("Test on", "user", m, "'s local test dataset, loss:", mean_loss, ", accuracy:", mean_accuracy)
        
        
        if final and not local_test:
            data_loader = make_data_loader({'test': dataset})['test']
            epoch_loss = []
            epoch_accuracy = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
                logger.append(evaluation, 'test', input_size)
                
                epoch_loss.append(evaluation["Global-Loss"])
                epoch_accuracy.append(evaluation["Global-Accuracy"])
                
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            mean_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
            #mean_compute_accuracy = sum(compute_acc) / len(compute_acc)
            print("Test on global model and global test dataset, loss:", mean_loss, ", accuracy:", mean_accuracy)
                
        if not final and not local_test:
            data_loader = make_data_loader({'test': dataset})['test']
            epoch_loss = []
            epoch_accuracy = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
                logger.append(evaluation, 'test', input_size)
                
                epoch_loss.append(evaluation["Global-Loss"])
                epoch_accuracy.append(evaluation["Global-Accuracy"])
                
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            mean_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
            #mean_compute_accuracy = sum(compute_acc) / len(compute_acc)
            print("Test on", "user", local_idx, "and global test dataset, loss:", mean_loss, ", accuracy:", mean_accuracy)
            
        if final:
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        else:
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                'Train Epoch: {}({:.0f}%)'.format(epoch, 100.), "Local Test"]}
        logger.append(info, 'test', mean=False)
        info = logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
        loss = float(info.split("Global-Loss:")[1].split("Global-Accuracy:")[0])
        accuracy = float(info.split("Global-Accuracy:")[1].split("Local Test")[0])
        
    return mean_loss, mean_accuracy


def make_local(dataset, data_split, label_split, federation):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    print("user_idx", user_idx)
    results["user_idx"] = user_idx
    results["model_rate"] = [federation.model_rate[user_idx[m]] for m in range(num_active_users)]
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(model_rate_m, data_loader_m, label_split[user_idx[m]])
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, data_loader, label_split):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        
        
    def train(self, local_parameters, lr, logger):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        #print(model)
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
        loss = []
        accuracy = []
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            epoch_loss = []
            epoch_accuracy = []
            #compute_acc = []
            for i, input in enumerate(self.data_loader):
                input = collate(input) 
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                #compute_acc.append(compute_accuracy(output["score"], input['label']))
                
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
                #print(local_epoch, i, cfg['metric_name']['train']['Local'], evaluation)
                epoch_loss.append(evaluation["Local-Loss"])
                epoch_accuracy.append(evaluation["Local-Accuracy"])
            
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            mean_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
            #mean_compute_accuracy = sum(compute_acc) / len(compute_acc)
            
            print("Local Epoch:", local_epoch, ", epoch loss:", mean_loss, ", epoch accuracy:", mean_accuracy)
            
        local_parameters = model.state_dict()
        return local_parameters, (mean_loss, mean_accuracy), model
    
    
def compute_accuracy(output, target):
    correct, total = 0, 0
    _, pred_label = torch.max(output.data, 1)
    #print(pred_label)
    total += output.data.size()[0]
    correct += (pred_label == target.data).sum().item()
    #print(correct/float(total))
    return correct/float(total)
    
    
if __name__ == "__main__":
    main()