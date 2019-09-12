import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import Config as config

def plot(path_top1, path_loss):
    max_epochs = config.no_of_epochs
    with open(path_top1, 'rb') as r:
        reward_data = pickle.load(r)
    with open(path_loss, 'rb') as l:
        loss_data = pickle.load(l)
    
    r_top1 = np.array(reward_data['r_top1'])
    r_loss = np.array(reward_data['r_loss'])
    r_epoch = np.array(reward_data['r_epoch'])
    l_top1 = np.array(loss_data['l_top1'])
    l_loss = np.array(loss_data['l_loss'])
    l_epoch = np.array(loss_data['l_epoch'])
    indeces = np.where(r_epoch < max_epochs)

    r_top1 = r_top1[indeces]
    l_top1 = l_top1[indeces]

    r_loss = r_loss[indeces]
    l_loss = l_loss[indeces]

    r_epoch = r_epoch[indeces]
    l_epoch = l_epoch[indeces]

    figure_loss, ax_loss = plt.subplots()
    figure_top1, ax_top1 = plt.subplots()

    ax_loss.plot(r_epoch, r_loss, label="train")
    ax_loss.plot(l_epoch, l_loss,'o', label="test")
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('loss')
    ax_loss.legend()
    ax_loss.set_title('Epoch vs Loss')

    ax_top1.plot(r_epoch, r_top1, label="train")
    ax_top1.plot(l_epoch, l_top1,'o', label="test")
    ax_top1.set_xlabel('Epochs')
    ax_top1.set_ylabel('top1')
    ax_top1.set_title('Epoch vs Top1')
    ax_top1.legend()

    figure_loss.savefig('plot_res/plot_loss.pdf')
    figure_top1.savefig('plot_res/plot_top1.pdf')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_vs_epoch', type=str, default='')
    parser.add_argument('--loss_vs_epoch', type=str, default='') 
    args = parser.parse_args()
    plot(args.top_vs_epoch, args.loss_vs_epoch) 

if __name__ == '__main__':
    main() 
