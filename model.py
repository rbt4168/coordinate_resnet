import torch
from torch import nn
import torchvision
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor

class GlobalModulation(nn.Module):
    def __init__(self, in_dim):
        super(GlobalModulation, self).__init__()
        self.fc = nn.Linear(in_dim, in_dim*2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        b, c, h, w = x.shape
        mod = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        mult, add = mod[:, :c], mod[:, c:]
        x = x * mult.reshape(b, c, 1, 1) + add.reshape(b, c, 1, 1)
        return x

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
    def __init__(self, out_dim=1, pretrained=False, fc_type="mlp1"):
        super(SpatialResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.avgpool = SpatialSoftmax()
        if fc_type == "mlp1":
            self.resnet18.fc = nn.Sequential(
                nn.Linear(1024, out_dim),
            )
        elif fc_type == "mlp2":
            self.resnet18.fc = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, out_dim)
            )
        elif fc_type == "mlpsquare":
            self.resnet18.fc = RootSquareMLP(1024, out_dim)
    def forward(self, x):
        x = self.resnet18(x)
        return x

class SpatialResNet18GlobalMod(nn.Module):
    def __init__(self, out_dim=1, pretrained=False, fc_type="mlp1"):
        super(SpatialResNet18GlobalMod, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.avgpool = nn.Sequential(
            GlobalModulation(512),
            SpatialSoftmax()
        )
        if fc_type == "mlp1":
            self.resnet18.fc = nn.Sequential(
                nn.Linear(1024, out_dim),
            )
        elif fc_type == "mlp2":
            self.resnet18.fc = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, out_dim)
            )
        elif fc_type == "mlpsquare":
            self.resnet18.fc = RootSquareMLP(1024, out_dim)
    def forward(self, x):
        x = self.resnet18(x)
        return x

class SpatialFPN(nn.Module):
    def __init__(self, out_dim=1, pretrained=False, fc_type="mlp1"):
        super(SpatialFPN, self).__init__()
        m = torchvision.models.resnet18(pretrained=pretrained)
        m.avgpool = nn.Identity()
        m.fc = nn.Identity()
        
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # dry run to get the number of channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.body(dummy)
        self.in_channels_list = [o.shape[1] for o in out.values()]

        self.fp_conv = nn.Sequential(
            nn.Conv2d(sum(self.in_channels_list), 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )
        
        self.pool = SpatialSoftmax()
        fc_in_dim = 512
        if fc_type == "mlp1":
            self.fc = nn.Sequential(
                nn.Linear(2*fc_in_dim, out_dim),
            )
        elif fc_type == "mlp2":
            self.fc = nn.Sequential(
                nn.Linear(2*fc_in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, out_dim)
            )
        elif fc_type == "mlpsquare":
            self.fc = RootSquareMLP(fc_in_dim, out_dim)

        else:
            raise ValueError(f"Unknown fc_type: {fc_type}")
        
        
    def forward(self, x):
        x = self.body(x)
        # upsample and concat features
        _, _, h, w = x["0"].shape
        layers = [nn.functional.interpolate(x[f"{i}"], size=(h, w), mode="bilinear") for i in range(4)]
        x = torch.cat(layers, dim=1)
        x = self.fp_conv(x)
        x = self.pool(x) # [b, c 2] -> [b, c*2]
        x = self.fc(x.reshape(x.shape[0], -1))
        return x
        
    
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
        model = SpatialResNet18(pretrained=pretrained, fc_type="mlpsquare")
    elif model_name == "spatial_mlp2_resnet18":
        model = SpatialResNet18(pretrained=pretrained, fc_type="mlp2")
    elif model_name == "resnet18":
        model = ResNet18(pretrained=pretrained)
    elif model_name == "spatial_fpn_resnet18":
        model = SpatialFPN(pretrained=pretrained, fc_type="mlp2")
    elif model_name == "spatial_global_resnet18":
        model = SpatialResNet18GlobalMod(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

if __name__ == "__main__":
    model = SpatialResNet18GlobalMod()
    dummy = torch.zeros(1, 3, 224, 224)
    out = model(dummy)
    print(out)
    