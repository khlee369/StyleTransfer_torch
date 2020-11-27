import torch
import torch.nn as nn
from torchvision.models import vgg19

'''
content = relu4_2
style = conv1_1 conv2_1 conv3_1 conv4_1 conv5_1
'''
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()