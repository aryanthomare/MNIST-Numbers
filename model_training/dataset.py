import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import numpy.typing as npt
from imageio.v2 import imread
from PIL import Image
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import pickle


def config(key: str) -> str:
    """Mock config function to return file paths."""
    paths = {
        "train_images_file": "model_training/mnist/train-images.idx3-ubyte",
        "train_labels_file": "model_training/mnist/train-labels.idx1-ubyte",
        "test_images_file": "model_training/mnist/t10k-images.idx3-ubyte",
        "test_labels_file": "model_training/mnist/t10k-labels.idx1-ubyte",
    }
    return paths[key]


data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((28, 28)),                 # Resize to 28x28
    transforms.RandomRotation(degrees=5),        # Small random rotation
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Normalize to [-1, 1]
])

train_img = config(f"train_images_file")  # e.g., "train-images.idx3-ubyte"
train_lbl = config(f"train_labels_file")  # e.g., "train-labels.idx1-ubyte"

test_img = config(f"test_images_file")  # e.g., "train-images.idx3-ubyte"
test_lbl = config(f"test_labels_file")  # e.g., "train-labels.idx1-ubyte"

import os
print(os.listdir())
X_train = idx2numpy.convert_from_file(train_img)  # Shape: (N, 28, 28)
y_train = idx2numpy.convert_from_file(train_lbl)  # Shape: (N,)


X_test = idx2numpy.convert_from_file(test_img)  # Shape: (N, 28, 28)
y_test = idx2numpy.convert_from_file(test_lbl)  # Shape: (N,)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.08, random_state=42)  # 0.25 x 0.8 = 0.2


class ImageStandardizer:
    """Standardize a batch of images to mean 0 and variance 1.

    The standardization should be applied separately to each channel.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """

    def __init__(self) -> None:
        """Initialize mean and standard deviations to None."""
        self.image_mean = None
        self.image_std = None

    def fit(self, X: npt.NDArray) -> None:
        """Calculate per-channel mean and standard deviation from dataset X."""
        # TODO: Complete this function
        self.image_mean = np.mean(X, axis=(0, 1, 2))
        self.image_std = np.std(X, axis=(0, 1, 2))
    
    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """Return standardized dataset given dataset X."""
        # TODO: Complete this function
        shift_mean = X - self.image_mean
        shift_std = shift_mean / self.image_std
        return shift_std
        

class NumberDataset(Dataset):
    """Dataset class for Number images."""

    def __init__(self, partition: str) -> None:
        """Read in the necessary data from disk.

        For parts 2 and 3, `task` should be "target".
        For source task of part 4, `task` should be "source".
        """
        super().__init__()
        self.partition = partition

        self.X, self.y = self._load_data()

    def _load_data(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Load a single data partition from MNIST .idx-ubyte files."""
        print(f"Loading {self.partition} data...")

        # Define file paths for images and labels
        
        image_file = config(f"{self.partition}_images_file")  # e.g., "train-images.idx3-ubyte"
        label_file = config(f"{self.partition}_labels_file")  # e.g., "train-labels.idx1-ubyte"



        # Load images and labels using idx2numpy
        X = idx2numpy.convert_from_file(image_file)  # Shape: (N, 28, 28)
        y = idx2numpy.convert_from_file(label_file)  # Shape: (N,)

        # Normalize images to [0, 1] range
        X = X.astype(np.float32) / 255.0

        return X, y

    def get_semantic_label(self, numeric_label: int) -> str:
        """Return the string representation of the numeric class label.

        (e.g., the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.y[numeric_label]
    
    def __len__(self) -> int:
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image, label) pair at index `idx` of dataset."""
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()


class NumberDatasetv2(Dataset):

    def __init__(self,X,y,train_transforms = False) -> None:
        """Read in the necessary data from disk.

        For parts 2 and 3, `task` should be "target".
        For source task of part 4, `task` should be "source".
        """
        super().__init__()

        self.X, self.y = X,y
        self.train_transforms = train_transforms

    def get_semantic_label(self, numeric_label: int) -> str:
        """Return the string representation of the numeric class label.

        (e.g., the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.y[numeric_label]
    

    def __len__(self) -> int:
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image, label) pair at index `idx` of dataset."""
        # Apply transformations to the image
        image = self.X[idx]
        if self.train_transforms:
            image = Image.fromarray(image.astype(np.uint8))  # Convert NumPy array to PIL image
            image = data_transforms(image)  # Apply transformations here
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.tensor(self.y[idx]).long()
        return image, label


def get_train_val_test_datasets(task: str = "default", **kwargs) -> tuple[NumberDatasetv2, NumberDatasetv2, NumberDatasetv2, ImageStandardizer]:
    """Return NumberDatasets and image standardizer.

    Image standardizer should be fit to train data and applied to all splits.
    """
    tr = NumberDatasetv2(X_train,y_train,train_transforms=True)
    va = NumberDatasetv2(X_val,y_val)
    te = NumberDatasetv2(X_test,y_test)



    # Resize
    # We don't resize images, but you may want to experiment with resizing
    # images to be smaller for the challenge portion. How might this affect
    # your training?
    # tr.X = resize(tr.X)
    # va.X = resize(va.X)
    # te.X = resize(te.X)

    # Standardize
    # standardizer = ImageStandardizer()
    
    # standardizer.fit(tr.X)

    # with open("image_standardizer.pkl", "wb") as f:
    #     pickle.dump(standardizer, f)

    # tr.X = standardizer.transform(tr.X)
    # va.X = standardizer.transform(va.X)
    # te.X = standardizer.transform(te.X)
    return tr, va, te


# idx = 1
# nums = NumberDataset("train")
# img = nums.__getitem__(idx)[0].numpy()
# print(f"Image shape: {img.shape}")
# imgplot = plt.imshow(img)
# plt.show()




def get_train_val_test_loaders(task: str, batch_size: int, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader, str]:
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the DogsDataset constructor.
    """
    tr, va, te = get_train_val_test_datasets(task, **kwargs)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader, tr.get_semantic_label


# te, va, tr = get_train_val_test_datasets(task="target", batch_size=64)
# img = te.__getitem__(0)[0].numpy()
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()