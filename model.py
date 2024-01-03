import torch
from torch import nn
import torchvision


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super(SpatialSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        u = torch.arange(h, dtype=torch.float32, device=x.device) / (h - 1)
        v = torch.arange(w, dtype=torch.float32, device=x.device) / (w - 1)
        mesh_u, mesh_v = torch.meshgrid(u, v)
        mesh_u = mesh_u.reshape(-1)
        mesh_v = mesh_v.reshape(-1)
        mesh = torch.stack((mesh_u, mesh_v), dim=1)
        mesh = mesh.reshape(1, 1, -1, 2)
        mesh = mesh.repeat(b, 1, 1, 1)
        x = x.reshape(b, c, -1)
        x = self.softmax(x).unsqueeze(-1)
        x = torch.sum(x * mesh, dim=2) # calculate the expected value of each channel
        # x: [b, c, 2] (A 2D coordinate for each channel)
        return x
    
class RootSquareMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RootSquareMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(), ## To make sure the output is positive
        )

    def forward(self, x):
        # x: [b (c 2)] -> [b, 2, c]
        x = x.reshape(x.shape[0], -1, 2).permute(0, 2, 1)
        x = self.mlp(x**2) # x: [b, 2, out_dim]
        x = torch.sum(x, dim=1) # x: [b, out_dim]
        return x**0.5
    
class SpatialResNet18(nn.Module):
    def __init__(self, out_dim=1, pretrained=False, square=False):
        super(SpatialResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.avgpool = SpatialSoftmax()
        self.resnet18.fc = nn.Sequential(
            nn.Linear(1024, out_dim)
        ) if not square else RootSquareMLP(512, out_dim)

    def forward(self, x):
        return self.resnet18(x)
    
class ResNet18(nn.Module):
    def __init__(self, out_dim=1, pretrained=False):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.resnet18(x)
    
def get_model(model_name, pretrained=False):
    if model_name == "spatial_resnet18":
        model = SpatialResNet18(pretrained=pretrained)
    elif model_name == "spatial_square_resnet18":
        model = SpatialResNet18(pretrained=pretrained, square=True)
    elif model_name == "resnet18":
        model = ResNet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model
    