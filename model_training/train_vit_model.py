import torch
import matplotlib.pyplot as plt
from dataset import get_train_val_test_loaders
from model_training.vitmodel import ViT
from train_common import count_parameters, restore_checkpoint, evaluate_epoch, train_epoch, save_checkpoint, early_stopping,make_training_plot
from utils import config, set_random_seed
    


def main():
    """Train ViT and show training plots."""
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=64
    )

    # TODO: Define the ViT Model according to the appendix D
    model = ViT(
        num_patches=7,
        num_blocks=4,
        num_hidden=6,
        num_heads=4,
        num_classes=10,
    )
    # for param in model.named_parameters():
    #     if param[1].requires_grad:
    #         print(param[0], param[1].shape)

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Number of float-valued parameters: {count_parameters(model)}")

    # Attempts to restore the latest checkpoint if exists
    print("Loading ViT...")
    model, start_epoch, stats = restore_checkpoint(model, "checkpoints")
    
    start_epoch = 0
    stats = []    

    axes = make_training_plot(name="ViT Training")

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        multiclass=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    patience = 5
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            multiclass=True,
            
        )
        

        save_checkpoint(model, epoch + 1, "checkpoints", stats)

        # Update early stopping parameters
        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")

    # Save figure and keep plot open; for debugging
    plt.savefig(f"vit_training_plot_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
