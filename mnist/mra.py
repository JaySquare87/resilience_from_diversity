

import argparse

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
alpha = 0.01


# %%
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import jaccard_score
from scipy.stats import pointbiserialr


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from tqdm.auto import tqdm

from torchattacks import PGD, FGSM, AutoAttack, MIFGSM

import os, sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from models.lenet5 import LeNet5

models_num = int(mnumber)
models = []
models_names = []
for i in range(models_num):
    models_names.append('./{}/lenet5{}.pth'.format(folder_name, i))
    models.append(None)



# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the network
for i in range(models_num):
    models[i] = LeNet5()

# Load the pretrained model
for i in range(models_num):
    models[i].load_state_dict(torch.load(models_names[i], map_location=device))

# Set the model in evaluation mode. In this case this is for the Dropout layers
for i in range(models_num):
    models[i].eval()


# %%
test_dataset = torchvision.datasets.MNIST(root='../data/mnist',
                                               train=False,
                                               download=True,
                                               transform=transforms.Compose([
                                                       transforms.Resize((32, 32)),
                                                       transforms.ToTensor()]))


# %%
test_loader = DataLoader(dataset=test_dataset,
                              batch_size=100,
                              shuffle=False)


# %%

loss = nn.CrossEntropyLoss()

pbar = tqdm(test_loader)

perturbed_images = []

for data, target in pbar:


    adv_model = np.random.randint(0,mnumber)
    
    if attack_to_use == 'fgsm':
        attack = FGSM(models[adv_model], eps=eps)
    elif attack_to_use == 'pgd':
        attack = PGD(models[adv_model], eps=eps, alpha=alpha, steps=40, random_start=True)
    elif attack_to_use == 'auto':
        attack = AutoAttack(models[adv_model], eps=eps)
    elif attack_to_use == 'mifgsm':
        attack = MIFGSM(models[adv_model], eps=eps)
    adv_image = attack(data, target)

    perturbed_images.append([adv_image, target])


# %%
temp = []
for batch, labels in perturbed_images:
    for i in range(len(batch)):
        temp.append([batch[i], labels[i]])


# %%
perturbed_images = temp


# %%
torch.save(perturbed_images, 'perturbed_dataset_{}_e{}.pt'.format(attack_to_use, eps))


# %%
perturbed_images = torch.load('perturbed_dataset_{}_e{}.pt'.format(attack_to_use, eps))


# %%
batch_size=100


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
col_str = []
col_str.append('target')
for i in range(len(models)):
    col_str.append('clm{}'.format(i))


# %%
df = pd.DataFrame(correct, columns=col_str)


# %%
accuracies = []
for _ in range(len(models)):
    acc = []
    for i in df.itertuples():
        if i.target == i[_+2]:
            acc.append(1)
        else:
            acc.append(0)
    accuracies.append(np.sum(np.array(acc))/len(df))
    print('Accuracy for submodel {} is: {}'.format(_+1, np.sum(np.array(acc))/len(df)))


# %%
print('MRA is : {}'.format(np.mean(accuracies)))
