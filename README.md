# Resilience from Diversity: Population-based approach to harden models against adversarial attacks


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run the following commands depending on the experiment:

```train
For the MNIST experiment:
python ./mnist/clm_train.py --folder <folder to save trained models> --nmodel <number of models to train> --alpha <alpha for counter linked training> --delta <delta for counter linked training> --pre <switch for training from pretrained models> --pref <folder for pretrained models if --pre switch is activated> --epochs <number of epochs to train> --prse <number of steps before PRSE is activated> --lr <learning rate> --adv <a switch whether this is an adversarial training or not>

For the CIFAR-10 experiment:
python ./cifar-10/clm_train.py --folder <folder to save trained models> --nmodel <number of models to train> --alpha <alpha for counter linked training> --delta <delta for counter linked training> --pre <switch for training from pretrained models> --pref <folder for pretrained models if --pre switch is activated> --epochs <number of epochs to train> --prse <number of steps before PRSE is activated> --lr <learning rate> --adv <a switch whether this is an adversarial training or not>
```

## Evaluation

To evaluate the models against adversarial attacks, run the following commands depending on the experiment:

```eval
For the MNIST experiment:
python ./mnist/mra.py --attack <attack to use> --folder <folder where the models are stored> --nmodel <number of models to use> --epsilon <perturbation magnitude> --testid <id for the test> --batch <batch size>

For the CIFAR-10 experiment:
python ./cifar-10/attack.py --attack <attack to use> --folder <folder where the models are stored> --nmodel <number of models to use> --epsilon <perturbation magnitude> --testid <id for the test> --batch <batch size>

The following is the list of attacks you can test against:
- fgsm: Fast Gradient Sign Method attack
- pgd: Projected Gradient Descent attack - Linf
- auto: AutoAttack
- mifgsm: MI-FGSM attack.
```

## Pre-trained Models

Pretrained models are included in the folders of mnist and cifar-10.

Since GitHub has a limit of the size of uploaded files, you can download the pretrained models through this link: https://drive.google.com/drive/folders/1Dkupi4bObIKofjKZOwOG0owsBFwfwo_5?usp=sharing

```
├── LICENSE
├── README.md
├── __init__.py
├── cifar-10
│   ├── clm10-a0.5d0.1-epochs150-prse10 <CLM with 10 submodels>
│   ├── clm_adv4-a0.1d0.05-epochs150-prse10 <CLM-Adv with 4 submodels>
│   ├── clm_train.py
│   ├── mra.py
│   ├── ulm10 <ULM with 10 submodels>
│   └── ulm_adv4 <ULM-Adv with 4 submodels>
├── mnist
│   ├── clm10-a0.1d0.1-epochs5-prse10 <CLM with 10 submodels>
│   ├── clm_adv4-a0.01d0.005-epochs5-prse1 <CLM-Adv with 4 submodels>
│   ├── clm_train.py
│   ├── mra.py
│   ├── ulm10 <ULM with 10 submodels>
│   └── ulm_adv4 <ULM-Adv with 4 submodels>
├── models
│   ├── lenet5.py
│   └── resnet.py
└── requirements.txt
```


## Contributing

MIT License
