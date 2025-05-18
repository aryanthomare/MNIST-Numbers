import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from convmodel import ConvModel
from dataset import get_train_val_test_loaders
import os
# Load the trained model

tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
    task="test",
    batch_size=10
)
print(os.listdir())

# TODO: Define the ViT Model according to the appendix D
checkpoint = torch.load('model_training/good_model/epoch=7.checkpoint.pth.tar', weights_only=False)


# model = ViT(
#     num_patches=4,
#     num_blocks= 2,
#     num_hidden=8,
#     num_heads=2,
#     num_classes=10,
# )

model = ConvModel()


model.load_state_dict(checkpoint['state_dict'])  # Load the model state dictionary
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for images, labels in te_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(f"Predicted: {predicted}, Actual: {labels}")


