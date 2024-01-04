import torch
from torch import nn
from torch.nn.functional import softmax
import torchvision
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange, repeat

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        # conv with group norm
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
        )
    def forward(self, x):
        return x + self.conv(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, stride=2),
            nn.GroupNorm(32, out_dim),
        )
    def forward(self, x):
        return self.downsample(x)
    
class SpatialAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(SpatialAttention, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(in_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x_in):
        # x: [b, c, h, w]
        b, c, h, w = x_in.shape
        x = self.in_conv(x_in)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attention(x, x, x)[0]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in
    
class MyResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MyResNet, self).__init__()
        # in_res = (224, 224)
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        ) # out_res = (56, 56)
        self.layer1 = nn.Sequential(
            GlobalModulation(64),
            ResidualBlock(64),
            ResidualBlock(64),  
        ) # out_res = (56, 56)
        self.down1 = DownBlock(64, 128) # out_res = (28, 28)
        self.layer2 = nn.Sequential(
            GlobalModulation(128),
            ResidualBlock(128),
            ResidualBlock(128),
            SpatialAttention(128),
        ) # out_res = (28, 28)
        self.down2 = DownBlock(128, 256) # out_res = (14, 14)
        self.layer3 = nn.Sequential(
            GlobalModulation(256),
            ResidualBlock(256),
            ResidualBlock(256),
            SpatialAttention(256),
        ) # out_res = (14, 14)
        self.down3 = DownBlock(256, 512) # out_res = (7, 7)
        self.layer4 = nn.Sequential(
            GlobalModulation(512),
            ResidualBlock(512),
            ResidualBlock(512),
            SpatialAttention(512),
        ) # out_res = (7, 7)
        self.spatial_softmax = SpatialSoftmax()
        self.out_fc = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layer1(x) + x
        x = self.down1(x)
        x = self.layer2(x) + x
        x = self.down2(x)
        x = self.layer3(x) + x
        x = self.down3(x)
        x = self.layer4(x) + x
        x = self.spatial_softmax(x).reshape(x.shape[0], -1)
        return self.out_fc(x)

class GlobalModulation(nn.Module):
    def __init__(self, in_dim):
        super(GlobalModulation, self).__init__()
        self.fc = nn.Linear(in_dim, in_dim*2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        b, c, h, w = x.shape
        mod = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        mult, add = mod[:, :c] + 1, mod[:, c:]
        x = x * mult.reshape(b, c, 1, 1) + add.reshape(b, c, 1, 1)
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(SpatialAttention, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(in_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [b, c, h, w]
        b, c, h, w = x.shape
        x = self.in_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attention(x, x, x)[0]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
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
        return x + 0.5

class HybridSpatialSoftmax(nn.Module):
    def __init__(self):
        super(HybridSpatialSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_head = nn.Conv2d(512, 512, 1)
        self.coordinate_embed = nn.Sequential(
            nn.Linear(2, 512),
            nn.GELU(),
        )
        self.semantic_head = nn.Sequential(
            nn.Conv2d(512, 1024, 1),
            nn.GELU(),
            nn.Conv2d(1024, 512, 1),
            nn.Sigmoid(),
        ) # [b, 1, 1, 1], semantic gating
        self.out_project = nn.Linear(512, 1)

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
        x1 = self.spatial_head(x).reshape(b, c, -1)
        x1 = self.softmax(x1).unsqueeze(-1)
        x1 = torch.sum(x1 * mesh, dim=2) # [b, c, 2] calculate the expected value of each channel
        g = self.semantic_head(self.avg_pool(x)).reshape(b, c, 1)
        return self.out_project(g * self.coordinate_embed(x1)).squeeze(-1) 
    
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
    def __init__(self, out_dim=1, pretrained=True, fc_type="mlp1"):
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

class HybridResNet18(nn.Module):
    def __init__(self, out_dim=1, pretrained=True, fc_type="mlp1"):
        super(HybridResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.avgpool = HybridSpatialSoftmax()
        if fc_type == "mlp1":
            self.resnet18.fc = nn.Sequential(
                nn.Linear(512, out_dim),
            )
        elif fc_type == "mlp2":
            self.resnet18.fc = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, out_dim)
            )
    def forward(self, x):
        x = self.resnet18(x)
        return x
    
class SpatialResNet18ColorGate(nn.Module):
    def __init__(self, out_dim=1, pretrained=True, fc_type="mlp1"):
        super(SpatialResNet18ColorGate, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.avgpool = SpatialSoftmax()
        # self.resnet18.fc = nn.Identity()
        
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
        black_pixels = torch.max(x, dim=1, keepdim=True)[0] < 0.5 # [b, 1, h, w]
        x = torch.abs(x - torch.mean(x, dim=(2, 3), keepdim=True)) # [b, c, h, w]
        x *= (1 - black_pixels.float())
        x = self.resnet18(x)
        return x

class SpatialResNet18Attn(nn.Module):
    def __init__(self, out_dim=1, pretrained=True, fc_type="mlp1"):
        super(SpatialResNet18Attn, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.avgpool = nn.Sequential(
            SpatialAttention(512),
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

class FPN(nn.Module):
    def __init__(self, out_dim=1, pretrained=False, fc_type="mlp1"):
        super(FPN, self).__init__()
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

        fc_in_dim = 960
        if fc_type == "mlp1":
            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, out_dim),
            )
        elif fc_type == "mlp2":
            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, out_dim)
            )
        else:
            raise ValueError(f"Unknown fc_type: {fc_type}")

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        
    def forward(self, x):
        x = self.body(x)
        # upsample and concat features
        _, _, h, w = x["0"].shape
        layers = [self.avgpool(x[f"{i}"]) for i in range(4)]
        x = torch.cat(layers, dim=1)
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
        model = SpatialResNet18ColorGate(pretrained=pretrained)
    elif model_name == "hybrid_mlp2_resnet18":
        model = HybridResNet18(pretrained=pretrained, fc_type="mlp2")
    elif model_name == "FPN_mlp2_resnet18":
        model = FPN(pretrained=pretrained, fc_type="mlp2")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

if __name__ == "__main__":
    model = SpatialResNet18ColorGate()
    dummy = torch.zeros(1, 3, 224, 224)
    out = model(dummy)
    print(out)
    