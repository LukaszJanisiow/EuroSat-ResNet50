from torch.utils.data import Dataset
from torchvision import transforms
import torch

transforms = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])

def unnormalize(tensor, mean, std):
    tensor = tensor.clone()  
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  
    return tensor

class EuroSat(Dataset):
    def __init__(self, data, transform=transforms):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['label']
        if self.transform:
            image = self.transform(image)
            
        return image, label