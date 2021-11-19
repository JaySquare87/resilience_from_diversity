import torch.nn as nn

class LeNet5(nn.Module):
    
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.mxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.mxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(120)

        self.fc = nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),
                nn.LogSoftmax(dim=-1))
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.mxpool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.mxpool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out