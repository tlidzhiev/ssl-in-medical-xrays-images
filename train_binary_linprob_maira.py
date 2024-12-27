import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Load model directly
from transformers import AutoImageProcessor, AutoModel

from src.trainer import Trainer, WandBWriter, Metrics_classification
from src.datasets import BinaryLabelDataset

class LinProbModel(nn.Module):
    def __init__(self, encoder, num_class=1) -> None:
        super().__init__()

        self.encoder = encoder
        self.fc = nn.Linear(768, num_class)
    
    def forward(self, x):
        x = self.encoder(**x)
        x = self.fc(x.pooler_output)
        return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', '--project_name', help='Project name', required=False,
                        type=str, default="train")
    parser.add_argument('-dir', '--data_dir_path', help='Directory path', required=False,
                        type=str, default="/kaggle/working/ssl-in-medical-xrays-images/data")
    return parser.parse_args()

def main():
    args = parse_args()

    writer = WandBWriter(
        project_name=args.project_name,
    )

    batch_size = 4 # len train_loader == 1000, can use with 16Gb GPU
    num_epochs = 20
    save_model_step = False
    scheduler_per_batch = True
    freeze_enc = True
    init_lr = 1e-3
    log_step = 250
    data_dir = args.data_dir_path

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino-maira-2")
    model = LinProbModel(AutoModel.from_pretrained("microsoft/rad-dino-maira-2"), 2).to(device)
    # print(model)

    # print(f"{data_dir}/dataset_256/train/images")
    train_dataset = BinaryLabelDataset(images_dir=f"{data_dir}/dataset_256/train/images", labels_dir=f"{data_dir}/dataset_256/train/labels", transform=processor)
    val_dataset = BinaryLabelDataset(images_dir=f"{data_dir}/dataset_256/val/images", labels_dir=f"{data_dir}/dataset_256/val/labels", transform=processor)
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    criterion = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=0.0001)

    metrics = Metrics_classification(num_classes=2, threshold=0.5, mode="binary")  # use  mode="binary" only in case of binary classification with 2 outputs 

    trainer = Trainer(
        num_epochs=num_epochs, 
        save_model_step=save_model_step, 
        scheduler_per_batch=scheduler_per_batch, 
        freeze_enc=freeze_enc, 
        log_step=log_step, 
        device = device,
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        scheduler=scheduler, 
        train_loader=train_loader,
        test_loader=val_loader, 
        writer=writer,
        metrics=metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
