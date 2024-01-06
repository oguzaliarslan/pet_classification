import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

# setting dataset for loader
class CatvsDogDataset(Dataset):
    def __init__(self, dataset, input_size):
        super().__init__()

        self.dataset = dataset
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        data_point, data_label = self.dataset[index]
        data_point = self.transform(data_point)

        # data labels are interchanged with the trained dataset
        return data_point, 1 - data_label

    def __len__(self):
        return len(self.dataset)


class AlexNetBackboned(nn.Module):
    def __init__(self):
        super(AlexNetBackboned, self).__init__()
        # Load pretrained alexnet
        self.alexnet = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", weights="DEFAULT")

        # freezing the alexnet parameters
        for param in self.alexnet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):


        x = self.alexnet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class VGG19BNBackboned(nn.Module):
    def __init__(self):
        super(VGG19BNBackboned, self).__init__()
        # Load pretrained vgg19_bn
        self.vgg19_bn = torch.hub.load("pytorch/vision:v0.10.0", "vgg19_bn", weights="DEFAULT")

        # freezing the vgg parameters
        for param in self.vgg19_bn.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):


        x = self.vgg19_bn(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class InceptionV3Backboned(nn.Module):
    def __init__(self):
        super(InceptionV3Backboned, self).__init__()
        # Load pretrained InceptionV3
        self.inceptionv3 = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", weights="DEFAULT")

        # some other thing related with inception model
        self.inceptionv3.aux_logits = False

        # freezing the inception parameters
        for param in self.inceptionv3.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.inceptionv3(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class DenseNetBackBoned(nn.Module):
    def __init__(self):
        super(DenseNetBackBoned, self).__init__()
        # Load pretrained densenet
        self.densenet = torch.hub.load("pytorch/vision:v0.10.0", "densenet161", weights="DEFAULT")

        # freezing the densenet parameters
        for param in self.densenet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):


        x = self.densenet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class ResnetBackBoned(nn.Module):
    def __init__(self):
        super(ResnetBackBoned, self).__init__()
        # Load pretrained resnet
        self.resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", weights="DEFAULT")

        # freezing the resnet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):


        x = self.resnet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class MobileNetV3Backboned(nn.Module):
    def __init__(self):
        super(MobileNetV3Backboned, self).__init__()
        # Load pretrained mobilenet
        self.mobilenet = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v3_large", weights="DEFAULT")## VGG19-BN

        # freezing the mobilenet parameters
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):


        x = self.mobilenet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x