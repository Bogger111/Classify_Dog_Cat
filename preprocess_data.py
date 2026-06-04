import os
import yaml
import torch
from torchvision import transforms
from torch.utils.data import random_split
from dataloader import CatDogDataset


def build_preprocess_transform(resize=260,cropsize=256):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
        )
    ])


def dataset_to_tensors(dataset):
    images = []
    labels = []

    for image,label in dataset:
        images.append(image)
        labels.append(label)

    return torch.stack(images), torch.stack(labels).long()


def save_tensor_file(path,dataset):
    images,labels = dataset_to_tensors(dataset)
    torch.save(
        {
            "images": images,
            "labels": labels
        },
        path
    )
    print(f"Saved {path}: images={tuple(images.shape)}, labels={tuple(labels.shape)}")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    train_dir = dataset_cfg["train_dir"]
    val_dir = dataset_cfg.get("val_dir")
    resize = dataset_cfg.get("resize",260)
    cropsize = dataset_cfg.get("cropsize",256)
    ratio = dataset_cfg.get("ratio",0.8)
    seed = config.get("train",{}).get("seed",42)
    output_dir = dataset_cfg.get("preprocessed_dir","data/preprocessed")

    os.makedirs(output_dir,exist_ok=True)
    transform = build_preprocess_transform(resize,cropsize)

    if val_dir is None:
        full_dataset = CatDogDataset(train_dir,transform=transform)
        train_size = int(ratio*len(full_dataset))
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset,val_dataset = random_split(
            full_dataset,
            [train_size,val_size],
            generator=generator
        )
    else:
        train_dataset = CatDogDataset(train_dir,transform=transform)
        val_dataset = CatDogDataset(val_dir,transform=transform)

    save_tensor_file(os.path.join(output_dir,"train.pt"),train_dataset)
    save_tensor_file(os.path.join(output_dir,"val.pt"),val_dataset)


if __name__ == "__main__":
    main()
