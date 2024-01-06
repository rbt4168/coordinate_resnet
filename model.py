import torch
from torch import nn
from torch.nn.functional import softmax
import torchvision
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange, repeat

class MyAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.drop_out = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        matrix_Q = self.w_q(query)
        matrix_K = self.w_k(key)
        matrix_V = self.w_v(value)

        matrix_Q = matrix_Q.view(batch_size, -1, self.n_heads, self.hid_dim ,self.n_heads).permute(0, 2, 1, 3)
        matrix_K = matrix_K.view(batch_size, -1, self.n_heads, self.hid_dim ,self.n_heads).permute(0, 2, 1, 3)
        matrix_V = matrix_V.view(batch_size, -1, self.n_heads, self.hid_dim ,self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(matrix_Q, matrix_K.permute(0, 1, 3, 2)) / self.scale
        attention = self.drop_out(torch.softmax(energy, dim=-1))

        x = torch.matmul(attention, matrix_V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        # conv with group norm
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=0),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=0),
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
        self.attention = nn.MultiheadAttention(in_dim, num_heads=num_heads, batch_first=True) # MyAttention(in_dim, num_heads, 0.2) 

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
            nn.Conv2d(3, 64, 7, stride=2, padding=0),
            nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=0),
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
            # SpatialAttention(128),
        ) # out_res = (28, 28)
        self.down2 = DownBlock(128, 256) # out_res = (14, 14)
        self.layer3 = nn.Sequential(
            GlobalModulation(256),
            ResidualBlock(256),
            ResidualBlock(256),
            # SpatialAttention(256),
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

# Define the basic residual block
class XResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(XResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output dimensions do not match, use a 1x1 convolutional layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)  # Add the residual
        out = self.relu(out)
        return out


class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Calculate query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))

        # Apply softmax to obtain attention weights
        attention_weights = self.softmax(scores)

        # Apply attention weights to the value
        attended_values = torch.matmul(attention_weights, value)

        return attended_values

class XSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(XSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, num_channels, height, width)
        out = self.gamma * out + x
        return out

class ResidualBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = XSelfAttention(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)  # Apply self-attention
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# Define the ResNet architecture
class XResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(XResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.normal_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.avgpool = nn.Sequential(
            XSelfAttention(512),
            SpatialSoftmax(),
        )
        # nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(1024, 512)

        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        ) 

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_n = self.normal_pool(x)
        x_n = x_n.reshape(x_n.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return self.fc2(torch.cat([x, x_n], dim=1))

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
    
class DoubleFeature(nn.Module):
    def __init__(self) -> None:
        super(DoubleFeature, self).__init__()

        # Load a pre-trained ResNet-18 model as the extractor
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.avgpool = nn.Sequential()
        self.resnet18.fc = nn.Sequential()

        # RNN layer
        # self.rnn = nn.LSTM(input_size=49, hidden_size=49, num_layers=3, batch_first=True)
        self.rnn = nn.RNN(input_size=49, hidden_size=49, num_layers=5, batch_first=True)

        self.spatial = SpatialSoftmax()

        # Contrastive loss layer
        self.fc = nn.Sequential(
            nn.Linear(512 *7 *7, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.resnet18(x)

        bsz = x.size(0)
        x = x.view(bsz, 512, -1)

        x, _ = self.rnn(x)

        x = x.view(bsz, 512, 7, 7)

        # Apply Spatial Softmax
        # x = self.spatial(x)
        # print(x.size(), bsz)
        x = x.reshape(bsz, -1)

        # Apply final fully connected layers
        x = self.fc(x)

        return x

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
    elif model_name == "testing":
        model = DoubleFeature()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

if __name__ == "__main__":
    model = DoubleFeature()
    dummy = torch.zeros(2, 3, 224, 224)
    out = model(dummy)
    print(out)
    