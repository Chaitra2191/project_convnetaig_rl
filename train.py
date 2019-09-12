import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import argparse
from torch.autograd import Variable
import time
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
import math
from torchvision import models
import pickle
#from ptflops import get_model_complexity_info
#from thop import profile

from convnetaig_rl import *
import Config as config

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset = [cifar10/cifar100/imagenet]')
parser.add_argument('--exp_name', default='exp1', type=str, help='Name of experiment')
parser.add_argument('--test_Only', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint (default: none), resume training from here')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Use visdom to track and plot')
parser.add_argument('--print_batchfreq', '-p', default=100, type=int, help='Print batch frequency (default: 10)')
parser.add_argument('--target_val', default=0.8, type=float, help='Target rate to compute target loss')
parser.add_argument('--loss_fact', default=2, type=float, help='Loss factor')
parser.add_argument('--pretrained', default='', type=str,help='Get the pretrained model')
args = parser.parse_args()
eps = np.finfo(np.float32).eps.item()
best_prediction = 0


def train(trainLoader, model, policy, criterion, optimizer, epoch, target_rates_gates, reward_data, plot_reward_data):
    policy.train()  # Policy
    model.train()  # Environment
    total_loss_values, top1_val, total_activations, total_batch_time, end_time = config.Initialize_Values()

    for i, (inputs, targets) in enumerate(trainLoader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        state = inputs.unsqueeze(1)
        for t in range(config.max_steps):
            action_prob = policy(state)
            action, logprobs = action_select(action_prob)
            output, activation_rates = model(inputs, action=action)
            # import pdb pdb.set_trace()
            # classification loss
            loss_classify = criterion(output, targets)

            # target rate loss
            activations = 0
            activations_plot = 0
            for j, value_act in enumerate(activation_rates):
                #if args.dataset == 'imagenet':
                    #print('TRinside', target_rates_gates[j])
                    #if target_rates_gates[j] < 1:
                        #activations_plot += torch.mean(value_act)
                        #activations += torch.pow(target_rates_gates[j] - torch.mean(value_act), 2)
                    #else:
                        #activations_plot += 1
                #else:
                activations_plot += torch.mean(value_act)
                activations += torch.pow(args.target_val - torch.mean(value_act), 2)
                # Save computation time and to minimize amt of unnecssary features
                # mean(act) fraction of time layer l is executed

            # Data Parallel mode
            activations_plot = torch.mean(activations_plot / len(activation_rates))
            activations = torch.mean(activations / len(activation_rates))

            # Loss factor- trade off term based on 2 loss terms
            activation_loss = args.loss_fact * activations

            # Reward - Maximize this
            total_loss = loss_classify + activation_loss

            if math.isnan(activations_plot.data):
                optimizer.zero_grad()
                total_loss.backward()
                continue

            policy.p_rewards.append(total_loss.data)
            policy.p_log_probs.append(logprobs)

        # Policy Gradient
        total_rewards = 0
        loss_policy = []
        l_rewards = []
        for reward_Val in policy.p_rewards[::-1]:
            total_rewards = reward_Val + config.gamma * total_rewards
            l_rewards.insert(0, total_rewards)
        l_rewards = torch.tensor(l_rewards)

        # Scale the Rewards
        l_rewards = (l_rewards - l_rewards.mean()) / (l_rewards.std() + eps)
        for log_prob, total_rewards in zip(policy.p_log_probs, l_rewards):
            loss_policy.append(-log_prob * total_rewards)

        loss_policy = torch.cat(loss_policy).sum()
        final_loss = loss_policy + total_loss

        # Compute the gradient
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        del policy.p_rewards[:]
        del policy.p_log_probs[:]

        # Measure the accuracy and record the loss
        precision_val = config.Compute_Accuracy(output.data, targets, topk_val=(1,5))[0]
        total_loss_values.val_update(final_loss.data, inputs.size(0))
        top1_val.val_update(precision_val, inputs.size(0))
        total_activations.val_update(activations_plot.data, 1)

        # Elapsed Time
        total_batch_time.val_update(time.time() - end_time)
        end_time = time.time()

        if i % args.print_batchfreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {total_batch_time.current_val:.3f} ({total_batch_time.average_val:.3f})\t'
                  'Loss: {total_loss_val.current_val:.4f} ({total_loss_val.average_val:.4f})\t'
                  'Prec@1: {top1_val.current_val:.3f} ({top1_val.average_val:.3f})\t'
                  'Activations Rates: {activation_val.current_val:.3f} ({activation_val.average_val:.3f})'.format(
                epoch, i, len(trainLoader), total_batch_time=total_batch_time, total_loss_val=total_loss_values,
                top1_val=top1_val, activation_val=total_activations))
    
    #Plot graphs
    if args.dataset == 'imagenet':
        plot_reward_data['r_epoch'].append(epoch)
        plot_reward_data['r_top1'].append(top1_val.average_val.item())
        plot_reward_data['r_loss'].append(total_loss_values.average_val.item())
        with open(reward_data, 'wb') as f:
            pickle.dump(plot_reward_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        if args.visdom:
            plotter.plot('top1', 'train', epoch, top1_val.average_val.item())
            plotter.plot('loss', 'train', epoch, total_loss_values.average_val.item())


def test(testLoader, model, policy, criterion, epoch, num_classes, target_rate_gates, loss_data, plot_loss_data):
    policy.eval()
    model.eval()

    total_loss_values, top1_val, total_activations, total_batch_time, end_time = config.Initialize_Values()
    
    if args.dataset == 'imagenet':
        gate_accumulation_rate = Gate_AccumulationRates_Imgnet(epoch=epoch, num_classes=num_classes)
    else:
        gate_accumulation_rate = Gate_AccumulationRates(epoch=epoch, num_classes=num_classes)

    for i, (inputs, targets) in enumerate(testLoader):
        with torch.no_grad():
            targets = targets.cuda()
            inputs = inputs.cuda()
            state = inputs.unsqueeze(1)
            for t in range(config.max_steps):
                # import pdb ; pdb.set_trace()
                action_prob = policy(state)
                action, _ = action_select(action_prob)
                output, activation_rates = model(inputs, action=action)

                activations = 0
                for j, value_act in enumerate(activation_rates):
                    #if args.dataset == 'imagenet':
                    #    if target_rate_gates[j] < 1:
                    #        activations += torch.mean(value_act)
                    #    else:
                    #        activations += 1
                    #else:
                    activations += torch.mean(value_act)

                # Data parallel
                activations = torch.mean(activations / len(activation_rates))
                if math.isnan(activations.data):
                    continue
                gate_accumulation_rate.gate_values(activation_rates, targets, num_classes, target_rate_gates)

            # Classification loss
            loss_classify = criterion(output, targets)

            # Measure accuracy and record loss
            precision_val = config.Compute_Accuracy(output.data, targets, topk_val=(1,))[0]
            total_loss_values.val_update(loss_classify.data, inputs.size(0))
            top1_val.val_update(precision_val, inputs.size(0))
            total_activations.val_update(activations.data, 1)

            # Elapsed time
            total_batch_time.val_update(time.time() - end_time)
            end_time = time.time()

            if i % args.print_batchfreq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time: {total_batch_time.current_val:.3f} ({total_batch_time.average_val:.3f})\t'
                      'Loss: {total_loss_val.current_val:.4f} ({total_loss_val.average_val:.4f})\t'
                      'Prec@1: {top1_val.current_val:.3f} ({top1_val.average_val:.3f})\t'
                      'Activations Rates: {activation_val.current_val:.3f} ({activation_val.average_val:.3f})'.format(
                    i, len(testLoader), total_batch_time=total_batch_time, total_loss_val=total_loss_values,
                    top1_val=top1_val, activation_val=total_activations))
        

        gate_output = gate_accumulation_rate.overall_result_accumulation()
        
        key_list = []
        val_list = []
        comp_saved = 0.0
        print('Activation Rates for Gates:')
        for dict_gateval in [gate_output[0]]:
            for key, val in dict_gateval.items():
            #for key, val in sorted(dict_gateval.items(), key=operator.itemgetter(1)):    
                dict_gateval[key] = round(val, 3)
                if epoch==350:
                    key_list.append(key)
                    val_list.append(val*100)
         
        print(dict_gateval)
        print('Precision@1: {top1_val.average_val:.3f}'.format(top1_val=top1_val))

        #Plot graphs
        if args.dataset == 'imagenet':
            plot_loss_data['l_epoch'].append(epoch)
            plot_loss_data['l_top1'].append(top1_val.average_val.item())
            plot_loss_data['l_loss'].append(total_loss_values.average_val.item())
            with open(loss_data, 'wb') as f:
                pickle.dump(plot_loss_data, f, pickle.HIGHEST_PROTOCOL)
        else:
            if args.visdom:
                plotter.plot('top1', 'test', epoch, top1_val.average_val.item())
                plotter.plot('loss', 'test', epoch, total_loss_values.average_val.item())

                if epoch % 5 == 0:
                    for gate in dict_gateval:
                        plotter.plot('frequency of execution', '{}'.format(gate), epoch, dict_gateval[gate])
        
            if epoch==350:
                comp_saved = config.plot_feq_gate(key_list, val_list)
                print('Computation Saved:', comp_saved)

                #if epoch % 5 == 0:
                   #for cat in gate_output[1]:
                       #plotter.plot('classes', '{}'.format(cat), epoch, gate_output[1][cat])

        return top1_val.average_val

def load_data(dataLoader):
    print('\nData Preparation')
    print("Preparing " + args.dataset + " dataset...")
    trainset = dataLoader(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = dataLoader(root='./data', train=False, download=False, transform=transforms.ToTensor())

    trainset.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.ComputeMeanStd(trainset, config.batch_size, 2)[0],
                             config.ComputeMeanStd(trainset, config.batch_size, 2)[1]),
    ])

    testset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.ComputeMeanStd(testset, config.batch_size, 2)[0],
                             config.ComputeMeanStd(testset, config.batch_size, 2)[1]),
    ])

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainLoader, testLoader

def load_data_imagenet(dataLoader):
    print('\nData Preparation for Imagenet')

    train_data = os.path.join('/netscratch/nayak/project/project-convnetaig-rl/imagenet/', 'train')
    val_data = os.path.join('/netscratch/nayak/project/project-convnetaig-rl/imagenet/', 'val')

    trainset = dataLoader(root=train_data, transform=transforms.ToTensor())
    testset = dataLoader(root=val_data, transform=transforms.ToTensor())
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    testset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=10, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=10, pin_memory=True)
    
    print('Data Preparation Completed')

    return trainLoader, testLoader

def load_checkpoint(model, policy):
    # Load checkpoint and resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint... '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            config.start_epoch = checkpoint['epoch']
            best_prediction = checkpoint['best_prediction']
            policy.load_state_dict(checkpoint['modelPolicy'])
            model.load_state_dict(checkpoint['modelEnv'])
            print("=> Loaded checkpoint! '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

def get_model(model):
    if args.dataset == 'imagenet':
        #if args.pretrained: #Uncomment this for regular model
        #Get the pretrained resnet50 for imagenet
        p_model = models.resnet50(pretrained=True)
        for param in p_model.parameters():
            param.requires_grad = False
        p_model_dictionary = p_model.state_dict() 
        model_dictionary = model.state_dict() 
        p_model_dictionary = {i: j for i, j in p_model_dictionary.items() if i in model_dictionary} 
        model_dictionary.update(p_model_dictionary)
        model.load_state_dict(model_dictionary)
        print('I am a pretrained model for imagenet')
    else:
        if args.pretrained:
            #Get the path of pretrained model for cifar
            checkpoint_pretrained = args.pretrained
            if os.path.isfile(checkpoint_pretrained):
                print("=> loading checkpoint '{}'".format(checkpoint_pretrained))
                checkpoint_data = torch.load(checkpoint_pretrained)
                current_state = model.state_dict()
                loaded_pretrained_state_dict = checkpoint_data
                for i in loaded_pretrained_state_dict:
                    if i in current_state:
                        current_state[i] = loaded_pretrained_state_dict[i]
                    else:
                        if 'fc' in i:
                            current_state[i.replace('fc', 'linear')] = loaded_pretrained_state_dict[i]
                        if 'downsample' in i:
                            current_state[i.replace('downsample', 'shortcut')] = loaded_pretrained_state_dict[i]
                model.load_state_dict(current_state)
                print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_pretrained, checkpoint_data['epoch']))
                print('I am a pretrained model for cifar')
            else:
                print("=> no checkpoint found at '{}'".format(checkpoint_pretrained))
                print('I am a pretrained model for cifar')
    
    print('I am a new model')
    return model

def main():
    global args, best_prediction
    args = parser.parse_args()

    manual_seed = 1
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    
    
    time_start = time.time()
    target_rates_gates = []
    if args.visdom:
        global plotter
        plotter = config.VisdomLinePlotter(env_name=args.exp_name)

    # Env and Policy CREATED and Load data
    policy = policy_network(args.dataset).cuda()
    
    #flops, params = get_model_complexity_info(policy, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
    #print('Flops',flops)
    #print('Param', params)
    
    if (args.dataset == 'cifar10'):
        num_classes = 10
        dataLoader = torchvision.datasets.CIFAR10
        model = env_resnet_cifar(110, 10).cuda()
        
        #flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        #flops, params = profile(model, inputs=(input, ))
        #print('Flops',flops)
        #print('Param', params)
        
        model = get_model(model)
        trainLoader, testLoader = load_data(dataLoader)
    elif (args.dataset == 'cifar100'):
        num_classes = 100
        dataLoader = torchvision.datasets.CIFAR100
        model = env_resnet_cifar(110, 100).cuda()
        model = get_model(model)
        trainLoader, testLoader = load_data(dataLoader)
    elif (args.dataset == 'imagenet'):
        num_classes = 1000
        dataLoader = torchvision.datasets.ImageFolder
        model = env_resnet_imagenet(50, 1000).cuda() #Change depth here
        model = get_model(model)
        trainLoader, testLoader = load_data_imagenet(dataLoader)
        #Set target rate for every layer, Default : same target rate for all
        #target_val_list = args.target_val * 64
        #target_rates_gates = {k: target_val_list[k] for k in range(len(target_val_list))}
    else:
        print('Please enter dataset name- cifar10/cifar100/imagenet')
        sys.exit()
    
    flop_model = model
    model = torch.nn.DataParallel(model).cuda()
    print("Total number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
     
    load_checkpoint(model, policy)

    cudnn.benchmark = True
    #cudnn.enabled = False
    #cudnn.deterministic = True
    
    criterion = nn.CrossEntropyLoss().cuda()
    params = list(policy.parameters()) + list(model.parameters())
    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    #optimizer = optim.SGD([{'params':params, 'lr':config.learning_rate, 'weight_decay':config.weight_decay, 'momentum':config.momentum}])
 
    #if not os.path.exists('data'):
    #    os.makedirs('data')
    
    reward_data = None
    loss_data = None
    #To plot imagenet data
    if args.dataset == 'imagenet':
        plot_reward_data = dict(r_top1=[], r_loss=[], r_epoch=[])
        plot_loss_data = dict(l_top1=[], l_loss=[], l_epoch=[])
        if reward_data and loss_data:
            reward_data = reward_data
            loss_data = loss_data
        else:
            reward_data = 'plot_res/top1_vs_epochs_'
            reward_data += time.strftime('%Y-%m-%d_%H-%M-%S') + '.pkl'
            loss_data = 'plot_res/loss_vs_epochs_'
            loss_data += time.strftime('%Y-%m-%d_%H-%M-%S')  + '.pkl'
    
    #Inference mode
    if args.test_Only:
        test_accuracy = test(testLoader, model, policy, criterion, config.no_of_epochs, num_classes, target_rates_gates, loss_data, plot_loss_data)
        sys.exit()

    for epoch in range(config.start_epoch, config.start_epoch + config.no_of_epochs):
        #Uncomment this for LR scheduler
        #config.lr_scheduler(optimizer, epoch, config.learning_rate)

        # Train the model
        train(trainLoader, model, policy, criterion, optimizer, epoch, target_rates_gates, reward_data, plot_reward_data)

        # Evaluate the model
        test_accuracy = test(testLoader, model, policy, criterion, epoch, num_classes, target_rates_gates, loss_data, plot_loss_data)

        # Save best model
        save_best_acc = test_accuracy > best_prediction
        best_prediction = max(test_accuracy, best_prediction)
        config.save_checkpoint({
            'epoch': epoch + 1,
            'modelPolicy': policy.state_dict(),
            'modelEnv': model.state_dict(),
            'best_prediction': best_prediction,
        }, save_best_acc, args.exp_name)
    print('Best Prediction: ', best_prediction)

    time_elapsed = time.time() - time_start
    print('| Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))

if __name__ == '__main__':
    main()

