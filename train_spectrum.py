import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import json
from glob import glob
from PIL import Image
from model import get_model
from accelerate import Accelerator
import yaml
import wandb
from ema_pytorch import EMA
import warnings

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root="", transform=None):
        self.root = root
        self.transform = transform
        picture_path = os.path.join(root, "pictures", "*.png")
        self.imgs = list(sorted(glob(picture_path)))
        with open("ans.json", "r") as f:
            self.ans = json.load(f)

    def __len__(self):
        return len(self.imgs)   
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            
        label_data = self.ans[img_path.split("/")[-1]]
        dist = ((label_data["x1"] - label_data["x2"])**2 + 
                (label_data["x2"] - label_data["y2"])**2)**0.5 

        return img.float(), float(dist)

class SpectrmuDataset(torch.utils.data.Dataset):
    def __init__(self, root="", transform=None):
        self.root = root
        self.transform = transform
        picture_path = os.path.join(root, "resized_spectrum/new_spectrum", "*.png")
        self.imgs = list(sorted(glob(picture_path)))
        with open(os.path.join(root, "label.json"), "r") as f:
            self.ans = json.load(f)

    def __len__(self):
        return len(self.imgs)   
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            
        label_data = self.ans[img_path.split("/")[-1]]

        return img.float()[:3], label_data

def train(args):
    cfg_name = args.cfg_name
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)[cfg_name]
    norm_label = config["normalize"]
    batch_size = config["bs"]
    epochs = config["epochs"]
    model_name = config["model"]
    dataset = config.get("dataset", "default")

    wandb.init(project="coordinate_resnet_spectrum", config=config)
    config = wandb.config

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if dataset == "default":
        dataset = MyDataset(transform=transform)
    elif dataset == "spectrum":
        dataset = SpectrmuDataset(root="../data", transform=transform)
    # 70% train, 10% valid, 20% test
    train_ind = int(len(dataset) * 0.7)
    valid_ind = int(len(dataset) * 0.8)
    train_dataset = Subset(dataset, range(train_ind))
    valid_dataset = Subset(dataset, range(train_ind, valid_ind))
    test_dataset = Subset(dataset, range(valid_ind, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = get_model(model_name)
    ema = EMA(model,
        beta=0.9995,
        update_every = 10,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    accelerator = Accelerator()
    model, optimizer, train_loader, valid_loader, test_loader, ema = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader, ema
    )

    best_valid_loss = float("inf")
    best_valid_L1_error = float("inf")  
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        with accelerator.autocast():
            device = accelerator.device
            for img, dist in tqdm(train_loader):
                img = img.to(device).float()
                dist = dist.to(device).float()
                if norm_label:
                    dist = (dist - 250) / 250
                optimizer.zero_grad()
                pred = model(img)
                loss = criterion(pred.squeeze(), dist)
                accelerator.backward(loss)
                optimizer.step()
                train_loss.append(loss.item())
                ema.update()
        print(f"Epoch {epoch} Train Loss: {sum(train_loss) / len(train_loss)}")
        wandb.log({"train_loss": sum(train_loss) / len(train_loss)})

        model.eval()
        ema.eval()
        valid_loss = []
        valid_L1_error = 0

        for img, dist in tqdm(valid_loader):
            img = img.to(device)
            if norm_label:
                dist = (dist - 250) / 250
            dist = dist.to(device)
            pred = ema(img)
            # pred = model(img)
            loss = criterion(pred.squeeze(), dist)
            dist_error = abs(pred.squeeze() - dist) 
            if norm_label:
                dist_error = dist_error * 250
            # count the number of correct predictions
            L1_error = dist_error.sum().item()
            valid_L1_error += L1_error
            valid_loss.append(loss.item())
        print(f"Epoch {epoch} Valid Loss: {sum(valid_loss) / len(valid_loss)}")
        print(f"Epoch {epoch} Valid L1 Error: {valid_L1_error / len(valid_dataset)}")
        
        wandb.log({"valid_loss": sum(valid_loss) / len(valid_loss)})
        wandb.log({"valid_L1_error": valid_L1_error / len(valid_dataset)})

        wandb.log({"best_valid_loss": best_valid_loss})
        wandb.log({"best_valid_L1_error": best_valid_L1_error})

        best_valid_loss = min(best_valid_loss, sum(valid_loss) / len(valid_loss))
        best_valid_L1_error = min(best_valid_L1_error, valid_L1_error / len(valid_dataset))

        if best_valid_loss == sum(valid_loss) / len(valid_loss):
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    # test  
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_loss = []
    test_L1_error = 0

    for img, dist in tqdm(test_loader):
        img = img.to(device)
        if norm_label:
            dist = (dist - 250) / 250
        dist = dist.to(device)
        pred = model(img)
        loss = criterion(pred.squeeze(), dist)
        dist_error = abs(pred.squeeze() - dist) 
        if norm_label:
            dist_error = dist_error * 250
        # count the number of correct predictions
        L1_error = dist_error.sum().item()
        test_L1_error += L1_error
        test_loss.append(loss.item())
    print(f"Test Loss: {sum(test_loss) / len(test_loss)}")
    print(f"Test L1 Error: {test_L1_error / len(test_dataset)}")
    wandb.log({"test_loss": sum(test_loss) / len(test_loss)})
    wandb.log({"test_L1_error": test_L1_error / len(test_dataset)})
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_name", type=str, default="base")
    args = parser.parse_args()
    train(args)
    