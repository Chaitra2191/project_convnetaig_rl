import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import numpy as np


def depth_resnet(depth):
    depth_list = [110, 50, 101, 152]
    if depth in depth_list:
        depth_resnet_dict = {
            '110': (BasicBlock, [18, 18, 18]),
            '50': (BottleneckBlock, [3, 4, 6, 3]),
            '101': (BottleneckBlock, [3, 4, 23, 3]),
            '152': (BottleneckBlock, [3, 8, 36, 3]),
        }
    else:
        print('Error : Resnet depth should be either 50, 101, 110, 152')

    return depth_resnet_dict[str(depth)]

#Define policy to sample actions to be passed into the resnet gate layers
class policy_network(nn.Module):
    def __init__(self, dataset):
        super(policy_network, self).__init__()
        if dataset == 'imagenet':
            self.fc1 = nn.Linear(150528, 64)
            nn.init.xavier_normal_(self.fc1.weight)
            self.fc2 = nn.Linear(64, 2)
            nn.init.xavier_normal_(self.fc2.weight)
        else:
            self.fc1 = nn.Linear(3072, 300)
            self.fc2 = nn.Linear(300, 2)

        self.p_log_probs = []
        self.p_rewards = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Action scores
        return F.softmax(x)

def action_select(action_prob):
    action_dist = Categorical(action_prob)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return (action, log_prob)

def conv3x3_cifar(input_planes, out_planes, stride=1):
    return nn.Conv2d(input_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

#Basic block of ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_cifar(input_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3_cifar(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

        #Two fully connected gate layers
        self.fc1 = nn.Conv2d(input_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm2d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2

    #Forward pass including computing gate relevance layer
    def forward(self, x, action):
        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.fc2(w)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out * w[:, 1].unsqueeze(1)
        out = F.relu(out)
        return out, w[:, 1]

#Bottleneck block of ResNet
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, input_planes, out_planes, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, self.expansion*out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

        #Two fully connected gate layers
        self.fc1 = nn.Conv2d(input_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm2d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2

    #Forward pass including computing gate relevance layer
    def forward(self, x, action):
        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1bn(self.fc1(w)))
        w = self.fc2(w)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.shortcut(x) + out * w[:, 1].unsqueeze(1)
        out = F.relu(out, inplace=True)
        return out, w[:, 1]

#Environment model (ResNet 110 for Cifar 10/100)
class env_resnet_cifar(nn.Module):
    def __init__(self, depth, num_of_classes):
        super(env_resnet_cifar, self).__init__()
        self.in_planes = 16
        basic_block, num_of_layers = depth_resnet(depth)

        self.conv1 = conv3x3_cifar(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(basic_block, 16, num_of_layers[0], 1)
        self.layer2 = self._make_layer(basic_block, 32, num_of_layers[1], 2)
        self.layer3 = self._make_layer(basic_block, 64, num_of_layers[2], 2)
        self.fc = nn.Linear(64 * basic_block.expansion, num_of_classes)
        self.ModelName = 'ResNet' + '_Cifar_' + str(num_of_classes)

        for i, j in self.named_modules():
            if isinstance(j, nn.Conv2d):
                if 'fc2' in str(i):
                    j.weight.data.normal_(0, 0.001)

    #To generate 17 basic blocks in each layer
    def _make_layer(self, basic_block, planes, num_of_layers, stride):
        strides = [stride] + [1] * (num_of_layers - 1)
        total_layers = []

        for stride_count in strides:
            total_layers.append(basic_block(self.in_planes, planes, stride_count))
            self.in_planes = planes * basic_block.expansion

        #To propagate the gating information into the layers of resnet. Also needed to compute the target loss for gates
        return Sequential_Extended_Gates(*total_layers)

    #Pass the action value to the layers and get the activation rates for gates
    def forward(self, x, action):
        activation_rates = []
        out = F.relu(self.bn1(self.conv1(x)))
        out, a = self.layer1(out, action)
        activation_rates.extend(a)
        out, a = self.layer2(out, action)
        activation_rates.extend(a)
        out, a = self.layer3(out, action)
        activation_rates.extend(a)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, activation_rates

#Environment model (ResNet 101 for imagenet)
class env_resnet_imagenet(nn.Module):
    def __init__(self, depth, num_of_classes):
        super(env_resnet_imagenet, self).__init__()
        self.in_planes = 64
        basic_block, num_of_layers = depth_resnet(depth)
                               
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(basic_block, 64, num_of_layers[0])
        self.layer2 = self._make_layer(basic_block, 128, num_of_layers[1], 2)
        self.layer3 = self._make_layer(basic_block, 256, num_of_layers[2], 2)
        self.layer4 = self._make_layer(basic_block, 512, num_of_layers[3], 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * basic_block.expansion, num_of_classes)
        self.ModelName = 'ResNet' + 'Imagenet ' + str(num_of_classes)

        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'fc2' in str(k):
                    m.weight.data.normal_(0, 0.001)

    #To generate 17 basic blocks in each layer
    def _make_layer(self, basic_block, planes, num_of_layers, stride=1):
        strides = [stride] + [1] * (num_of_layers - 1)
        total_layers = []

        for stride_count in strides:
            total_layers.append(basic_block(self.in_planes, planes, stride_count))
            self.in_planes = planes * basic_block.expansion

        # o propagate the gating information into the layers of resnet. Also needed to compute the target loss for gates
        return Sequential_Extended_Gates(*total_layers)

    #Pass the action value to the layers and get the activation rates for gates
    def forward(self, out, action):
        activation_rates = []
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out, a = self.layer1(out, action)
        activation_rates.extend(a)
        out, a = self.layer2(out, action)
        activation_rates.extend(a)
        out, a = self.layer3(out, action)
        activation_rates.extend(a)
        out, a = self.layer4(out, action)
        activation_rates.extend(a)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, activation_rates

class Sequential_Extended_Gates(nn.Module):
    def __init__(self, *args):
        super(Sequential_Extended_Gates, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module_val in args[0].items():
                self.add_module(key, module_val)
        else:
            for idx, module_val in enumerate(args):
                self.add_module(str(idx), module_val)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input, action, openings=None):
        activation_rates = []
        for i, module_val in enumerate(self._modules.values()):
            input, activation_rate = module_val(input, action)
            activation_rates.append(activation_rate)

        return input, activation_rates


class Gate_AccumulationRates_Imgnet():
    def __init__(self, epoch, num_classes):
        self.number_of_classes = {k: 0 for k in range(num_classes)}
        self.num_of_batches = 0
        self.epoch = epoch
        self.number_of_blocks = [3,4,6,3] #Change number of block here
        self.total_gates = {k: 0 for k in range(np.sum(self.number_of_blocks))}

    def gate_values(self, activations, labels, num_classes, target_rates_gates):
        for l, activation_val in enumerate(activations):
            if target_rates_gates[l] < 1:
                self.total_gates[l] += torch.sum(activation_val)
            else:
                self.total_gates[l] += labels.size(0)

            if self.epoch % 5 == 0:
                for m in range(num_classes):
                    if target_rates_gates[l] < 1:
                        self.number_of_classes[m] += torch.sum(activation_val[labels == m]).data
                    else:
                        self.number_of_classes[m] += torch.sum(labels == m).data

            self.num_of_batches += 1

    def overall_result_accumulation(self):
        if self.epoch % 5 == 0:
            return ([{m: self.total_gates[m].data.cpu().numpy() / 5000 for m in self.total_gates},
                    {m: self.number_of_classes[m].data.cpu().numpy() / 5000 / np.sum(self.number_of_blocks) for m in self.number_of_classes}])
        else:
            return ([{m: self.total_gates[m].data.cpu().numpy() / 5000 for m in self.total_gates}])

class Gate_AccumulationRates():
    def __init__(self, epoch, num_classes):
        self.number_of_classes = {k: 0 for k in range(num_classes)}
        self.num_of_batches = 0
        self.epoch = epoch
        self.number_of_blocks = [18, 18, 18]
        self.total_gates = {k: 0 for k in range(np.sum(self.number_of_blocks))}

    def gate_values(self, activations, labels, num_classes,target_rate_gates):
        for l, activation_val in enumerate(activations):
            self.total_gates[l] += torch.sum(activation_val)

            if self.epoch % 5 == 0:
                for m in range(num_classes):
                    self.number_of_classes[m] += torch.sum(activation_val[labels == m])
            self.num_of_batches += 1

    def overall_result_accumulation(self):
        if self.epoch % 5 == 0:
            return ([{m: self.total_gates[m].data.cpu().numpy() / 1000 for m in self.total_gates},
                    {m: self.number_of_classes[m].data.cpu().numpy() / 100 / np.sum(self.number_of_blocks) for m in self.number_of_classes}])
        else:
            return ([{m: self.total_gates[m].data.cpu().numpy() / 1000 for m in self.total_gates}])

if __name__ == "__main__":
    modelList = [env_resnet_cifar(110, 10), env_resnet_cifar(110, 100), env_resnet_imagenet(50, 1000), env_resnet_imagenet(101, 1000), env_resnet_imagenet(152, 1000)]
    for model in modelList:
        print('Resnet model:', model)
        print(model.ModelName)
        #print("Total number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        # print("Total number of layers:", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, model.parameters()))))
