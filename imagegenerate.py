import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")

def plot_train_val_curves(train_metrics, val_metrics, metric_name="Loss", save_path=None):

    epochs = np.arange(1, len(train_metrics) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_metrics, label=f"Train {metric_name}", marker='o')
    plt.plot(epochs, val_metrics, label=f"Val {metric_name}", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Train vs Val {metric_name}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, save_path=None):

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if class_names is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(label) for label in labels]
    else:
        labels = np.arange(len(class_names))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_sample_predictions(images, labels, preds=None, class_names=None, n=8, save_path=None):

    images = images.cpu()
    labels = labels.cpu()
    if preds is not None:
        preds = preds.cpu()

    n = min(n, images.shape[0])
    plt.figure(figsize=(n*2, 2.5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        img = images[i].permute(1,2,0).numpy()  # C,H,W -> H,W,C
        img = (img * 0.5) + 0.5  # 反归一化
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        label = int(labels[i].item())
        title = f"T:{class_names[label]}" if class_names else f"T:{label}"
        if preds is not None:
            pred = int(preds[i].item())
            title += f"\nP:{class_names[pred]}" if class_names else f"\nP:{pred}"
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
