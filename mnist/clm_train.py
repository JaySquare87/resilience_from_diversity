import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

import numpy as np
from tqdm.auto import tqdm
import csv

from torchattacks import PGD

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse

parser = argparse.ArgumentParser(description='CLM training for the CIFAR-10 experiment')
parser.add_argument('--adv', help='Adversarial Training? 1 - Yes, 0 - No', required=True, default=0)
parser.add_argument('-f','--folder', help='Folder to save the trained networks in', required=True)
parser.add_argument('-n', '--nmodel', help='Number of sub-models to train', required=True, default=4)
parser.add_argument('-a', '--alpha', help='Alpha Hyper-parameter', default=0.1)
parser.add_argument('-d', '--delta', help='Delta Hyper-parameter', default=0.1)
parser.add_argument('--clm', help='Counter-link train the networks', required=True, default=1)
parser.add_argument('--pre', help='Load from a pre-trained models', default=0)
parser.add_argument('--pref', help='Folder of pre-trained models', default='.')
parser.add_argument('--epochs', help='# of epochs to train the models', default=10)
parser.add_argument('--prse', help='Periodic Random Similarity Examination every n step', default=10)
parser.add_argument('--lr', help='Learning rate', default=0.001)



args = vars(parser.parse_args())

from models.lenet5 import LeNet5

folder_name = './mnist/'+args['folder']+args['nmodel']+'-a'+args['alpha']+'d'+args['delta']+'-epochs'+args['epochs']+'-prse'+args['prse']+'/'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

torch.autograd.set_detect_anomaly(True)

batch_size = 100
filter = 1
c_o = 5
num_classes = 10

# CLM hyper parameters
models_num = int(args['nmodel'])
models = []
best_acc = []
criterions = []
optimizers = []
losses = []
nodes = []
for i in range(models_num):
    models.append(None)
    best_acc.append(0)
    criterions.append(None)
    optimizers.append(None)
    losses.append(None)
    nodes.append(None)
mask = np.random.randint(2, size=(filter, c_o, c_o))
if int(args['clm']) == 1:
    mask_reshaped = mask.reshape(mask.shape[0], -1)
    # Save the mask for future analysis
    np.savetxt(folder_name+'mask.csv', mask_reshaped, delimiter=',')

flinks = mask.flatten()
for i in range(flinks.shape[0]):
    if flinks[i] == 1:
        n_monitor = i
        break
for i in range(flinks.shape[0]):
    if flinks[i] == 0:
        n_monitor0 = i

nwriter = csv.writer(open(folder_name+'node_analysis.csv', 'w'), delimiter=',')

link = False
start = True

alpha = float(args['alpha'])
delta = float(args['delta'])
prse = int(args['prse'])

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training Hyper parameters
learning_rate= float(args['lr'])#0.001
num_epochs = int(args['epochs'])


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/mnist', 
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),
                                                   transforms.ToTensor()]))

test_dataset = torchvision.datasets.MNIST(root='./data/mnist', 
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),
                                                   transforms.ToTensor()]))

# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False)

# Setup Models
for i in range(models_num):
    models[i] = LeNet5().to(device)

# Loss and optimizer
for i in range(models_num):
    criterions[i] = nn.CrossEntropyLoss()
    optimizers[i] = optim.Adam(models[i].parameters(), lr=learning_rate)


node_values = []
for i in range(models_num):
    node_values.append(0)

# Check if train from pre-trained models
if int(args['pre']) == 1:
    models_names = []
    for i in range(models_num):
        models_names.append('./mnist/{}/lenet5{}.pth'.format(args['pref'], i))
        models.append(None)
    for i in range(models_num):
        models[i].load_state_dict(torch.load(models_names[i], map_location=device))

total_steps = 0

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i in range(models_num):
        models[i].train()
        PRSEcounter = 0
        pbar = tqdm(train_loader)
        for idx, (images, labels) in enumerate(pbar):
            total_steps += 1
            PRSEcounter += 1
            images = images.to(device)
            labels = labels.to(device)
            
            model1 = i
            model2 = np.random.randint(0, models_num)
            while model1 == model2:
                model2 = np.random.randint(0, models_num)
            
            # Do PRSE
            if int(args['clm']) == 1:
                if PRSEcounter == prse: # and epoch > 0:
                    PRSEcounter = 0
                    with torch.no_grad():
                        nodes[model1] = models[model1].conv1.weight.cpu().detach().numpy()
                        nodes[model2] = models[model2].conv1.weight.cpu().detach().numpy()
                        diff = nodes[model1] - nodes[model2]
                        Beta = np.where(np.abs(diff) < delta, 1, 0)
                        O = np.zeros([filter, c_o, c_o])
                        nweight = (O + (-1)**model1 * alpha) * Beta * mask + nodes[model1]
                        nweight = nweight.astype(np.float32)
                        nweight = nn.Parameter(torch.from_numpy(nweight)).cuda()
                        for w in range(models[model1].conv1.weight.shape[0]):
                            models[model1].conv1.weight[w] = nweight[w]

            
            if args['adv'] == '1':
                attack = PGD(models[model1], eps=0.3, alpha=0.01, steps=100, random_start=True)
                images = attack(images, labels)

            # Forward pass
            outputs = models[model1](images)
            losses[i] = criterions[i](outputs, labels)

            # Monitor the selected node value
            for j in range(models_num):
                fweight = models[j].conv1.weight[0].cpu().detach().numpy().flatten()
                node_values[j] = fweight[n_monitor]

            
            # build writer dictionary
            writer_dic = {}
            for m in range(models_num):
                writer_dic['model-{}'.format(m)] = node_values[m]

            

            writer.add_scalars('node', writer_dic,
                total_steps
            )
            
            # Backward and optimize
            optimizers[i].zero_grad()
            losses[i].backward(retain_graph=True)
            optimizers[i].step()
            
            
            pbar.set_description('Model {} | Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f}'.format(model1+1, epoch+1, num_epochs, idx+1, total_step, losses[i].item()))
            
            for model in range(models_num):
                nodes[model] = models[model].conv1.weight.cpu().detach().numpy()

        # Test the model
        models[i].eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = models[model1](images)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += torch.eq(predicted, labels).float().sum().item()

            acc = 100.0 * correct / total
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))
            torch.save(models[model1].state_dict(), '{}lenet5{}.pth'.format(folder_name, model1))
