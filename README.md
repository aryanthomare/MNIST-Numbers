MNIST Digit Classifier — CNN & ViT
===================================

This project provides a complete training and visualization pipeline for classifying handwritten digits using Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). It includes live training graphs, model saving and loading, and an interactive drawing tool for real-time inference.

Overview
--------
This project trains and compares two separate neural network architectures on the MNIST dataset:

- A custom CNN model
- A Vision Transformer (ViT) model

Additional features include:
- Live training plots (loss and accuracy)
- Automatic model checkpoint saving
- Easy model reloading for inference or continued training
- An interactive drawing interface where you can draw a digit and view the model's prediction in real time

Features
--------
Training:
- Full training loop for both CNN and ViT
- Automatic MNIST downloading and preprocessing
- Configurable hyperparameters

Live Visualization:
- Real-time Matplotlib graphs
- Tracks training and validation loss and accuracy

Model Saving:
- Saves model weights, optimizer state, and training metadata
- Reloadable for inference or continued training

Drawing Inference Tool:
- Screen-based drawing interface
- Real-time digit prediction
- Automatic resizing, grayscaling, and normalization

License
-------
This project is intended for educational and research purposes. Modification and reuse are permitted.
