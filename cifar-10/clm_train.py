import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torchattacks import PGD


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

import numpy as np
from tqdm.auto import tqdm
import csv

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
parser.add_argument('--lr', help='Learning rate', default=0.1)


args = vars(parser.parse_args())

folder_name = args['folder']+args['nmodel']+'-a'+args['alpha']+'d'+args['delta']+'-epochs'+args['epochs']+'-prse'+args['prse']

from models.resnet import ResNet18

if not os.path.exists('./cifar-10/{}/'.format(folder_name)):
    os.makedirs('./cifar-10/{}'.format(folder_name))

torch.autograd.set_detect_anomaly(True)

# Hyper Parameters
models_num = int(args['nmodel'])
filter = 64
c_o = 3

nodes = []
models = []
best_acc = []  # best test accuracy
criterions = []
optimizers = []
schedulers = []
losses = []
for i in range(models_num):
    criterions.append(None)
    optimizers.append(None)
    schedulers.append(None)
    losses.append(None)
    models.append(None)
    best_acc.append(0)
    nodes.append(None)
mask = np.random.randint(2, size=(c_o, c_o, c_o))
if int(args['clm']) == 1:
    mask_reshaped = mask.reshape(mask.shape[0], -1)
    # Save the mask for future analysis
    np.savetxt(open(folder_name+'mask.csv', 'w'), mask_reshaped, delimiter=',')

flinks = mask.flatten()
for i in range(flinks.shape[0]):
    if flinks[i] == 1:
        n_monitor = i
        break
for i in range(flinks.shape[0]):
    if flinks[i] == 0:
        n_monitor0 = i

nwriter = csv.writer(open('./cifar-10/'+folder_name+'/node_analysis.csv', 'w'), delimiter=',')

link = False
start = True

alpha = float(args['alpha'])
delta = float(args['delta'])
prse = int(args['prse'])


'''Train CIFAR10 with PyTorch.'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=200, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
for i in range(models_num):
    models[i] = ResNet18()
    models[i] = models[i].to(device)

for i in range(models_num):
    models[i] = torch.nn.DataParallel(models[i])#.to(device)

# Check if train from pre-trained models
if int(args['pre']) == 1:
    models_names = []
    for i in range(models_num):
        print('Loading pre-trained model {}'.format(i+1))
        models_names.append('./cifar-10/{}/resnet{}.pth'.format(args['pref'], i))
    for i in range(models_num):
        models[i].load_state_dict(torch.load(models_names[i])['net']) #, map_location=device



for i in range(models_num):
    criterions[i] = nn.CrossEntropyLoss()
    optimizers[i] = optim.SGD(models[i].parameters(), lr=float(args['lr']), momentum=0.9, weight_decay=5e-4)
    schedulers[i] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[i], T_max=200)

node_values = []
for i in range(models_num):
    node_values.append(0)



total_steps = 0

# Training
epochs = int(args['epochs'])
total_step = len(trainloader)
for epoch in range(epochs):
    for i in range(models_num):
        PRSEcounter = 0
        models[i].train()
        correct = 0
        total = 0
        pbar = tqdm(trainloader)
        for idx, (inputs, targets) in enumerate(pbar):
            total_steps += 1
            PRSEcounter += 1
            inputs, targets = inputs.to(device), targets.to(device)

            optimizers[i].zero_grad()

            model1 = i
            model2 = np.random.randint(0, models_num)
            while model2 == model1:
                model2 = np.random.randint(0, models_num)

            # PRSE
            if int(args['clm']) == 1:
                if PRSEcounter == prse: #and epoch > 0:
                    PRSEcounter = 0
                    with torch.no_grad():
                        nodes[model1] = models[model1].module.conv1.weight.cpu().detach().numpy()
                        nodes[model2] = models[model2].module.conv1.weight.cpu().detach().numpy()
                        diff = nodes[model1] - nodes[model2]
                        Beta = np.where(np.abs(diff) < delta, 1, 0)
                        O = np.zeros([filter, c_o, c_o, c_o])
                        nweight = (O + (-1)**model1 * alpha) * Beta * mask + nodes[model1]
                        nweight = nweight.astype(np.float32)
                        nweight = nn.Parameter(torch.from_numpy(nweight)).cuda()
                        for w in range(models[model1].module.conv1.weight.shape[0]):
                            models[model1].module.conv1.weight[w] = nweight[w]

            if args['adv'] == '1':
                attack = PGD(models[model1], eps=8/255, alpha=2/255, steps=20, random_start=True)
                inputs = attack(inputs, targets)
            
            outputs = models[i](inputs)


            losses[i] = criterions[i](outputs, targets)
            losses[i].backward()
            optimizers[i].step()

            # Monitor the selected node value
            for j in range(models_num):
                fweight = models[j].module.conv1.weight[0].cpu().detach().numpy().flatten()
                node_values[j] = fweight[n_monitor]

            
            # build writer dictionary
            writer_dic = {}
            for m in range(models_num):
                writer_dic['model-{}'.format(m)] = node_values[m]

            

            writer.add_scalars('node', writer_dic,
                total_steps
            )


            pbar.set_description('Model {} | Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f}'.format(model1+1, epoch+1, epochs, idx+1, total_step, losses[i].item()))
        
            
            for model in range(models_num):
                nodes[model] = models[model].module.conv1.weight.cpu().detach().numpy()


        # Test
        models[i].eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = models[i](inputs)
                loss = criterions[i](outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                    
        acc = 100.0 * correct / total
        print('Test Accuracy of model {} on the 10000 test images: {} %'.format(i+1, acc))
        state = {
                'net': models[i].state_dict(),
                'acc': acc,
                'epoch': epoch,
                }
        torch.save(state, './cifar-10/{}/resnet{}.pth'.format(folder_name, i))
        schedulers[i].step()

