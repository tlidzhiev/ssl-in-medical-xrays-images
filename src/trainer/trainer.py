import torch
import numpy as np
import tqdm

class Trainer():
    def __init__(self, num_epochs, save_model_step, scheduler_per_batch, freeze_enc, log_step, device,
                 model, optimizer, criterion, scheduler, train_loader, test_loader, writer):
        self.num_epochs = num_epochs
        self.save_model_step = save_model_step
        self.scheduler_per_batch = scheduler_per_batch
        self.freeze_enc = freeze_enc
        self.log_step = log_step

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.writer = writer

        self.device = device

    def training_epoch(self, epoch, tqdm_desc):
        self.writer.set_step(epoch - 1)
        self.writer.add_scalar("epoch", epoch)

        train_loss = []

        self.model.train()
        if self.freeze_enc:
            self.model.eval()
            for param in self.model.fc.parameters(): # or change to your head
                param.requires_grad = True

        for batch in tqdm(self.train_loader, desc=tqdm_desc):
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()
            if self.scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()

            train_loss.append(loss.item())

        self.writer.set_step(epoch - 1)
        self.writer.add_scalar(
            "learning rate", self.scheduler.get_last_lr()[0]
        )
        self.writer.add_scalar("train loss", np.mean(train_loss))

    @torch.no_grad()
    def validation_epoch(self, epoch, part, tqdm_desc):
        test_loss = []

        self.model.eval()

        for images, labels in tqdm(self.test_loader, desc=tqdm_desc):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            test_loss.append(loss.item())

        self.writer.set_step(epoch - 1, part)
        self.writer.add_scalar(f"{part} loss", np.mean(test_loss))

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.training_epoch(
                epoch,
                tqdm_desc=f'Training {epoch}/{self.num_epochs}'
            )

            self.validation_epoch(
                epoch, "val",
                tqdm_desc=f'Validating {epoch}/{self.num_epochs}'
            )

            if self.scheduler is not None:
                if not self.scheduler_per_batch:
                    self.scheduler.step()
            
            if epoch % self.save_model_step == 0:
                torch.save(self.model.state_dict(), "weights.pt")
