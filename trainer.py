import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

CRITERION_DICT = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss
}

OPTIMIZER_DICT = {
    "Adam": optim.Adam,
    "SGD": optim.SGD
}

SCHEDULER_DICT = {
    "StepLR": optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR
}

class Trainer:
    def __init__(self,model,train_loader,val_loader,config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_cfg = config.get("train", {})
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = train_cfg.get("cudnn_benchmark", True)
        self.use_amp = train_cfg.get("amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        crit_cfg = config.get("criterion", {"type":"CrossEntropyLoss", "params":{}})
        self.criterion = CRITERION_DICT[crit_cfg["type"]](**crit_cfg.get("params", {}))

        opt_cfg = config.get("optimizer", {"type":"Adam"})
        opt_class = OPTIMIZER_DICT[opt_cfg["type"]]
        if opt_cfg["type"] == "Adam":
            self.optimizer = opt_class(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay",0))
        elif opt_cfg["type"] == "SGD":
            self.optimizer = opt_class(model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg.get("momentum",0), weight_decay=opt_cfg.get("weight_decay",0))
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg['type']}")

        sched_cfg = config.get("scheduler", None)
        if sched_cfg:
            sched_class = SCHEDULER_DICT[sched_cfg["type"]]
            self.scheduler = sched_class(self.optimizer, **{k:v for k,v in sched_cfg.items() if k!="type"})
        else:
            self.scheduler = None

        self.best_acc = 0
        self.currentepoch = 0
        self.checkpoint_dir = config.get('checkpoint_dir','./checkpoints')
        os.makedirs(self.checkpoint_dir,exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self):
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(self.train_loader,desc="Training",leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(x)
                loss = self.criterion(pred, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()*y.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

        if self.scheduler:
            self.scheduler.step()

        return total_loss / total, correct / total
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation", leave=False):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                total_loss += loss.item() * y.size(0)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        loss = total_loss / total

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_checkpoint("best")

        return loss, acc

    def _get_checkpoint_name(self,tag=""):
        
        opt = self.config["optimizer"]["type"]
        lr = self.config["optimizer"].get("lr", 0)
        crit = self.config["criterion"]["type"]
        sched = self.config.get("scheduler", {}).get("type", "None")
        
        filename =  f"{opt}_lr{lr}_{crit}_{sched}_{tag}.pth"

        return filename

    def save_checkpoint(self, tag):
        filename = self._get_checkpoint_name(tag)
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": self.currentepoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "best_acc": self.best_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "config": self.config
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.use_amp and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        
        self.best_acc = checkpoint.get("best_acc", 0)
        self.currentepoch = checkpoint.get("epoch",0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_accs = checkpoint.get("train_accs", [])
        self.val_accs = checkpoint.get("val_accs", [])

        start_epoch = self.currentepoch + 1

        print(
            f"Checkpoint loaded from {path}, "
            f"best_acc={self.best_acc:.4f}, "
            f"start_epoch={start_epoch}"
        )

        return start_epoch

    def get_all_val_preds(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    preds = self.model(x).argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y)
        return torch.cat(all_preds), torch.cat(all_labels)

    def fit(self, epochs,class_names=None,start_epoch = 1):
        for epoch in range(start_epoch, epochs + 1):
            self.currentepoch = epoch

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Best Acc: {self.best_acc:.4f}")

            self.save_checkpoint('last')

        from imagegenerate import plot_train_val_curves, plot_confusion_matrix, plot_sample_predictions
        output_dir = self.config.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        plot_train_val_curves(
            self.train_losses,
            self.val_losses,
            "Loss",
            save_path=os.path.join(output_dir, "loss_curve.png"),
        )
        plot_train_val_curves(
            self.train_accs,
            self.val_accs,
            "Accuracy",
            save_path=os.path.join(output_dir, "accuracy_curve.png"),
        )

        all_preds, all_labels = self.get_all_val_preds()
        plot_confusion_matrix(
            all_labels,
            all_preds,
            class_names=class_names,
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

        images, labels = next(iter(self.val_loader))
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            preds = self.model(images.to(self.device, non_blocking=True)).argmax(dim=1)
        plot_sample_predictions(
            images,
            labels,
            preds,
            class_names=class_names,
            save_path=os.path.join(output_dir, "sample_predictions.png"),
        )
