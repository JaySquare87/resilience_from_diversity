#%%
import argparse

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

parser = argparse.ArgumentParser(description='CLM training for the CIFAR-10 experiment')
parser.add_argument('--attack', help='Attack type', required=True, default=0)
parser.add_argument('-f','--folder', help='Folder of pretrained models - Has to be inside the "mnist" folder', required=True)
parser.add_argument('-n', '--nmodel', help='Number of sub-models to attack', required=True, default=4)
parser.add_argument('-e', '--epsilon', help='Perturbation magnitude epsilon', required=True, default=0.3)
parser.add_argument('--testid', help='Set the test id', required=False, default='0')
parser.add_argument('--batch', help='Set the batch size', required=False, default='100')
args = vars(parser.parse_args())


# %%
folder_name = args['folder']
mnumber = args['nmodel']
attack_to_use = args['attack']
eps = float(args['epsilon'])
alpha = 2/255
batch_size = int(args['batch'])

from models.resnet import ResNet18

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pandas as pd

from sklearn.metrics import jaccard_score


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.nn.functional as F

import numpy as np
from tqdm.auto import tqdm

from torchattacks import PGD, FGSM, AutoAttack, FAB, MIFGSM



# %%
models_num = int(mnumber)
models = []
models_names = []
for i in range(models_num):
    models_names.append('./{}/resnet{}.pth'.format(folder_name, i))
    models.append(None)


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = "cuda" if (torch.cuda.is_available()) else "cpu"

# Initialize the network
for i in range(models_num):
    models[i] = ResNet18()


for i in range(models_num):
    models[i] = torch.nn.DataParallel(models[i]).cuda()


# Load the pretrained model
for i in range(models_num):
    models[i].load_state_dict(torch.load(models_names[i], map_location=device)['net'])


# Set the model in evaluation mode. In this case this is for the Dropout layers
for i in range(models_num):
    models[i].eval()


# %%
transform_test = transforms.Compose([
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# %%
loss = nn.CrossEntropyLoss()

pbar = tqdm(test_loader)

perturbed_images = []

counter = 0

for data, target in pbar:
        
    adv_model = np.random.randint(0,mnumber)

    if attack_to_use == 'fgsm':
        attack = FGSM(models[adv_model], eps=eps)
    elif attack_to_use == 'pgd':
        attack = PGD(models[adv_model], eps=eps, alpha=alpha, steps=40, random_start=True)
    elif attack_to_use == 'auto':
        attack = AutoAttack(models[adv_model], eps=eps)
    elif attack_to_use == 'fab':
        attack = FAB(models[adv_model], eps=eps)
    elif attack_to_use == 'mifgsm':
        attack = MIFGSM(models[adv_model], eps=eps)
    adv_image = attack(data, target)

    perturbed_images.append([adv_image, target])



# %%
temp = []
for batch, labels in perturbed_images:
    for i in range(len(batch)):
        #temp.append([torch.reshape(batch[i], (3,32,32)), labels[i]])
        temp.append([batch[i], labels[i]])


# %%
perturbed_images = temp


# %%
torch.save(perturbed_images, 'perturbed_dataset_{}_e{:.3f}.pt'.format(attack_to_use, eps))


# %%
perturbed_images = torch.load('perturbed_dataset_{}_e{:.3f}.pt'.format(attack_to_use, eps))


# %%
perturbed_dataset = torch.utils.data.DataLoader(dataset = perturbed_images, batch_size=batch_size, shuffle=True)


# %%
correct = []
    
for adv_image, target in tqdm(perturbed_dataset):
    mcorrect = []
    for model in models:
        output = model(adv_image)
        pred = np.argmax(output.cpu().detach().numpy(), axis=1)
        mcorrect.append(pred)
    for batch in range(batch_size):
        temp = []
        temp.append(target[batch].item())
        for i in range(len(models)):
            temp.append(mcorrect[i][batch])
        correct.append(temp)


# %%
columns_lst = []
columns_lst.append('target')
for i in range(models_num):
    columns_lst.append('clm{}'.format(i+1))


# %%
columns_lst


# %%
df = pd.DataFrame(correct, columns=columns_lst)


# %%
df.head()


# %%
model_accuracy = []
for i in range(models_num):
    correct = []
    for index, row in df.iterrows():
        if row['clm{}'.format(i+1)] == row['target']:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = np.sum(correct)/len(df)
    model_accuracy.append(accuracy)
    print('Accuracy for submodel {} is: {}'.format(i+1, np.sum(np.array(accuracy))/len(df)))



# %%
for m in model_accuracy:
    print(m)


# %%
print('MRA is : {}'.format(np.mean(model_accuracy)))


