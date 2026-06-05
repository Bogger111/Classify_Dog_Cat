import os

import matplotlib

matplotlib.use("Agg")

import torch
import yaml

from dataloader import get_dataloaders
from imagegenerate import (
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_train_val_curves,
)
from Resnet import MyResNet18
from trainer import Trainer


CLASS_NAMES = ["cat", "dog"]


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    _, val_loader = get_dataloaders(
        train_dir=dataset_cfg["train_dir"],
        val_dir=dataset_cfg["val_dir"],
        batch_size=dataset_cfg["batch_size"],
        num_workers=dataset_cfg.get("num_workers", 2),
    )

    model = MyResNet18(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    )
    trainer = Trainer(model, None, val_loader, config)

    checkpoint_path = config["train"].get("resume_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(
            config.get("checkpoint_dir", "./checkpoints"),
            "Adam_lr0.001_CrossEntropyLoss_StepLR_best.pth",
        )
    trainer.load_checkpoint(checkpoint_path)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    loss_path = os.path.join(output_dir, "loss_curve.png")
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    if trainer.train_losses and trainer.val_losses:
        plot_train_val_curves(
            trainer.train_losses,
            trainer.val_losses,
            "Loss",
            save_path=loss_path,
        )
        print(f"Saved {loss_path}")
    else:
        print("Skipped loss curve: checkpoint has no saved loss history")

    if trainer.train_accs and trainer.val_accs:
        plot_train_val_curves(
            trainer.train_accs,
            trainer.val_accs,
            "Accuracy",
            save_path=acc_path,
        )
        print(f"Saved {acc_path}")
    else:
        print("Skipped accuracy curve: checkpoint has no saved accuracy history")

    all_preds, all_labels = trainer.get_all_val_preds()
    confusion_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names=CLASS_NAMES,
        save_path=confusion_path,
    )

    images, labels = next(iter(val_loader))
    images_device = images.to(trainer.device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=trainer.use_amp):
        preds = trainer.model(images_device).argmax(dim=1).cpu()

    samples_path = os.path.join(output_dir, "sample_predictions.png")
    plot_sample_predictions(
        images,
        labels,
        preds,
        class_names=CLASS_NAMES,
        save_path=samples_path,
    )

    print(f"Saved {confusion_path}")
    print(f"Saved {samples_path}")


if __name__ == "__main__":
    main()
