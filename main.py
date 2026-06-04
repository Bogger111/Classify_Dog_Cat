import yaml
import os
import torch
from net import MyResNet18
from dataloader import get_dataloaders,get_tensor_dataloaders
from trainer import Trainer

if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    dataset_cfg = config['dataset']
    preprocessed_dir = dataset_cfg.get(
        'preprocessed_dir',
        os.path.join(dataset_cfg.get('local_ssd_data_root','data'),'preprocessed')
    )
    train_pt = os.path.join(preprocessed_dir,'train.pt')
    val_pt = os.path.join(preprocessed_dir,'val.pt')
    num_workers = dataset_cfg.get('num_workers',2)
    use_preprocessed = dataset_cfg.get('use_preprocessed', False)

    if use_preprocessed and os.path.exists(train_pt) and os.path.exists(val_pt):
        train_data = torch.load(train_pt,map_location='cpu')
        val_data = torch.load(val_pt,map_location='cpu')
        train_loader ,val_loader = get_tensor_dataloaders(
            train_images=train_data['images'],
            train_labels=train_data['labels'],
            val_images=val_data['images'],
            val_labels=val_data['labels'],
            batch_size=dataset_cfg['batch_size'],
            num_workers=dataset_cfg.get('tensor_num_workers', 0)
        )
        print(f"Using preprocessed tensor files from {preprocessed_dir}")
    else:
        train_loader ,val_loader = get_dataloaders(
            train_dir=dataset_cfg['train_dir'],
            val_dir=dataset_cfg["val_dir"],
            batch_size=dataset_cfg['batch_size'],
            num_workers=num_workers
        )
        print("Using image files directly")

    model = MyResNet18(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )

    trainer = Trainer(model,train_loader,val_loader,config)
    
    start_epoch = 1
    resume_path = config['train'].get('resume_path', None)

    if resume_path and os.path.exists(resume_path):
        start_epoch = trainer.load_checkpoint(resume_path)
    
    else:
        print('No checkpoint found, traning from scratch')

    try:
        trainer.fit(config['train']['epochs'],start_epoch=start_epoch)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
