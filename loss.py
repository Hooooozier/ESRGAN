import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import config


class VGGLoss(nn.Module):
    def __init__(self, model_name: str='SRGAN'):
        super().__init__()
        if model_name == "SRGAN":
            self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        else:
            # self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval().to(config.DEVICE)
            self.vgg = vgg19(pretrained=True).features[:35].eval().to(config.DEVICE)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, x, target):
        x_features = self.vgg(x)
        target_feature = self.vgg(target)
        return self.loss(x_features, target_feature)
