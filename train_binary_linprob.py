import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Load model directly
from transformers import AutoImageProcessor, AutoModel

from src.trainer import Trainer, WandBWriter
from src.datasets import BinaryLabelDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', '--project_name', help='Project name', required=False, type=str, default="train")
    return parser.parse_args()

def main():
    args = parse_args()

    writer = WandBWriter(
        project_name=args.project_name,
    )

    batch_size = 64
    num_epochs = 100
    save_model_step = True
    scheduler_per_batch = True
    freeze_enc = False
    init_lr = 1e-3

    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    print(model)

    train_dataset = BinaryLabelDataset(images_dir=, labels_dir=, transform=T.ToTensor())
    val_dataset = BinaryLabelDataset(images_dir=, labels_dir=, transform=T.ToTensor())
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    log_step = len(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    criterion = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=0.0001)

    trainer = Trainer(
        num_epochs=num_epochs, 
        save_model_step=save_model_step, 
        scheduler_per_batch=scheduler_per_batch, 
        freeze_enc=freeze_enc, 
        log_step=log_step, 
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        scheduler=scheduler, 
        train_loader=train_loader,
        test_loader=val_loader, 
        writer=writer
    )

    trainer.train()

if __name__ == "__main__":
    main()
