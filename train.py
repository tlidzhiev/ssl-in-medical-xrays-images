import argparse

from src.trainer import Trainer, WandBWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', '--project_name', help='Project name', required=False, type=str, default="train")
    return parser.parse_args()

def main():
    args = parse_args()

    writer = WandBWriter(
        project_name=args.project_name,
    )

    model = 
    print(model)

    optimizer = 

    criterion = 

    scheduler = 

    train_loader = 

    num_epochs = 100
    save_model_step = True
    scheduler_per_batch = True
    freeze_enc = False
    log_step = len(train_loader)

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
        test_loader=None, 
        writer=writer
    )

    trainer.train()

if __name__ == "__main__":
    main()
