import os
import shutil
import torch
from visdom import Visdom
import numpy as np
import time
import matplotlib.pyplot as plt

#Hyperparameters
start_epoch = 1
no_of_epochs = 350
batch_size = 64
learning_rate = 0.001
momentum = 0.9
weight_decay = 5e-4
max_steps = 15
gamma = 0.97

def lr_scheduler(optimizer, epoch, lr):
    initial_lr = 0.001
    if epoch >= 150:
        lr = initial_lr  * lr
    if epoch >= 250:
        lr = initial_lr * lr
    #print('LR', lr)
    optimizer.param_groups[0]['lr'] = lr

#Mean and Std of datasets
def ComputeMeanStd(dataset, bs, num_workers):
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    meanVal = 0.
    stdVal = 0.
    for data, _ in dataLoader:
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        meanVal += data.mean(2).sum(0)
        stdVal += data.std(2).sum(0)
    
    meanVal /= len(dataLoader.dataset)
    stdVal /= len(dataLoader.dataset)

    return meanVal, stdVal


#Save checkpoint to resume training
def save_checkpoint(state, save_best_Acc, exp_name, file_name='checkpoint.pth.tar'):
    directory = "runs/%s/" % (exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = directory + file_name
    torch.save(state, file_name)
    if save_best_Acc:
        shutil.copyfile(file_name, 'runs/%s/' % (exp_name) + 'best_model.pth.tar')

#Plot data using Visdom
class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.scatter(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name],
                                 name=split_name,update='append')

#To calculate current and average values
class Compute_Avg_Sum_Values(object):
    def __init__(self):
        self.reset_values()

    def reset_values(self):
        self.total_sum = 0
        self.current_val = 0
        self.total_count = 0
        self.average_val = 0

    def val_update(self, current_val, k=1):
        self.total_sum += current_val * k
        self.current_val = current_val
        self.total_count += k
        self.average_val = self.total_sum / self.total_count

#Compute top precision at k
def Compute_Accuracy(outputs, labels, topk_val=(1,)):
    maxk_val = max(topk_val)
    batch_size = labels.size(0)

    _, prediction = outputs.topk(maxk_val, 1, True, True)
    prediction = prediction.t()
    correct_val = prediction.eq(labels.view(1, -1).expand_as(prediction))

    accuracy = []
    for k in topk_val:
        correctk_val = correct_val[:k].view(-1).float().sum(0)
        accuracy.append(correctk_val.mul_(100.0 / batch_size))
    return accuracy

def Initialize_Values():
    total_loss_values = Compute_Avg_Sum_Values()
    top1_val = Compute_Avg_Sum_Values()
    total_activations = Compute_Avg_Sum_Values()
    total_batch_time = Compute_Avg_Sum_Values()
    end_time = time.time()

    return total_loss_values, top1_val, total_activations, total_batch_time, end_time

def plot_feq_gate(key_list, val_list):
    y_pos = np.arange(len(key_list))
    plt.figure(figsize=(18,4))
    plt.bar(y_pos, val_list)
    plt.xticks(y_pos, key_list)
    plt.xlabel('Gates')
    plt.ylabel('Frequency of Execution (%)')
    plt.savefig('./plot_res/avg_execution_rate_per_gate')
    plt.show()

    comp_saved = "{0:.2f}".format(100 - sum(val_list)/len(val_list)) + "%"

    return comp_saved

