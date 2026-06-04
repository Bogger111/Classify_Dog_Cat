import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset,Subset,TensorDataset
import torch

def get_transform(resize=60,cropsize=256,train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(cropsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5]
            )
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5]
            )
        ])

    return transform

class CatDogDataset(Dataset):
    def __init__(self,root_dir,transform=None,color="RGB"):
        super().__init__()

        self.root = root_dir
        self.transform = transform
        self.color = color

        self.image_names = sorted([
            name for name in os.listdir(root_dir)
            if name.lower().endswith('.jpg')
        ])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.root,image_name)

        image = Image.open(image_path).convert(self.color)

        if image_name.startswith("cat"):
            label = 0
        elif image_name.startswith("dog"):
            label = 1
        else:
            raise ValueError(f'Unknown label:{image_name}')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(label,dtype=torch.long)

def get_dataloaders(
        train_dir,val_dir=None,batch_size=32,ratio=0.8,
        resize=260,cropsize=256,num_classes=2,seed=42,num_workers=2
        ):
    
    train_full_dataset = CatDogDataset(
        train_dir,
        transform=get_transform(resize,cropsize,train=True)
        )
    
    if val_dir is None:
        
        val_full_dataset = CatDogDataset(
        train_dir,
        transform=get_transform(resize,cropsize,train=False)
        )

        dataset_size = len(train_full_dataset)
        train_size = int(ratio*dataset_size)
        val_siaze = dataset_size - train_size

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_size,generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(train_full_dataset,train_indices)
        val_dataset = Subset(val_full_dataset,val_indices)

    else:
        train_dataset =train_full_dataset
        val_dataset = CatDogDataset(
            val_dir,
            transform=get_transform(resize,cropsize,train=False)
        )
    
    pin_memory = torch.cuda.is_available()
    train_dataloader = DataLoader(
        train_dataset,batch_size=batch_size,
        shuffle=True,num_workers=num_workers,pin_memory=pin_memory,
        persistent_workers=num_workers > 0
        )
    val_dataloader = DataLoader(
        val_dataset,batch_size=batch_size,
        shuffle=False,num_workers=num_workers,pin_memory=pin_memory,
        persistent_workers=num_workers > 0
        )
    
    return train_dataloader,val_dataloader

def get_tensor_dataloaders(
        train_images,train_labels,val_images,val_labels,batch_size=32,num_workers=0
        ):
    
    train_dataset = TensorDataset(train_images,train_labels)
    val_dataset = TensorDataset(val_images,val_labels)

    pin_memory = torch.cuda.is_available()
    train_dataloader = DataLoader(
        train_dataset,batch_size=batch_size,
        shuffle=True,num_workers=num_workers,pin_memory=pin_memory,
        persistent_workers=num_workers > 0
        )
    val_dataloader = DataLoader(
        val_dataset,batch_size=batch_size,
        shuffle=False,num_workers=num_workers,pin_memory=pin_memory,
        persistent_workers=num_workers > 0
        )
    
    return train_dataloader,val_dataloader
    
