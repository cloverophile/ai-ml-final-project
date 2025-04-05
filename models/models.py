import torch.nn as nn
import torchvision.models as models

def get_resnet18_pretrained(num_classes, transfer_learning=True):
    model = models.resnet18(pretrained=transfer_learning)
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vgg16_pretrained(num_classes, transfer_learning=True):
    model = models.vgg16(pretrained=transfer_learning)
    if transfer_learning:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model    

def get_resnet18_simple(num_classes, transfer_learning=False):
    model = models.resnet18(pretrained=False)  # Do not load pre-trained weights
    '''if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers if transfer learning '''
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vgg16_simple(num_classes, transfer_learning=False):
    model = models.vgg16(pretrained=False)  # Do not load pre-trained weights
    ''''if transfer_learning:
        for param in model.features.parameters():
            param.requires_grad = False  # Freeze all layers of 'features' if transfer learning'''
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model