import pygame
import sys
import random
import pygame
from pygame.locals import *
import math
from pygame import gfxdraw
from copy import deepcopy
import numpy as np
import pygame.freetype  # Import the freetype module.
from model import ConvModel
import torch
import torch.nn as nn
import pickle
import idx2numpy
from torchvision import transforms
import pygame.surfarray
import matplotlib.pyplot as plt

white = (255,255,255)
green = (165,238,93)
pink = (245,212,229)
red = (230,69,83)
blue = (85,75,230)
black = (0,0,0)
grey = (130,130,130)


pygame.init()
pygame.font.init() # you have to call this at the start, 
my_font = pygame.font.SysFont('Comic Sans MS', 20)

size_ratio = 30
img_size = 28
width, height = img_size*size_ratio, img_size*size_ratio

screen = pygame.display.set_mode((width,height))
fpsClock = pygame.time.Clock()


def backg1():
    screen.fill(black)



model = ConvModel()

# checkpoint = torch.load("draw_test//vit_txt.checkpoint.pth.tar",weights_only=False)
checkpoint = torch.load("model_training//good_model//epoch=3.checkpoint.pth.tar",weights_only=False)

model.load_state_dict(checkpoint['state_dict'])
model.eval()


from sklearn import preprocessing as p 
  
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert NumPy array to PyTorch tensor

    transforms.Resize((28, 28)) , # Resize to 28x28
    # transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # Normalize to [-1, 1]
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Normalize to [-1, 1]

])


pred_lbl = -1
def model_predict(img_in):
    img_tensor = transform(img_in).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        print(output)
        _, predicted = torch.max(output.data, 1)

        predicted = predicted.item()
        return predicted
        # Draw th+e text on the screen



while True:
    xcord, ycord = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == K_c:
                backg1()
            # if event.key == K_m:
            #     screen_array = pygame.surfarray.array3d(screen)

            #     # Transpose the array to (height, width, 3)
            #     screen_array = screen_array.transpose((1, 0, 2))

            #     # Convert to grayscale by averaging the RGB values
            #     grayscale_array = screen_array.mean(axis=2)

            #     # Normalize the grayscale array to [0, 1]
            #     grayscale_array = grayscale_array / 255.0

            #     # Convert to float32 to match model input type
            #     grayscale_array = grayscale_array.astype(np.float32)
                
            #     pred_lbl = model_predict(grayscale_array)
                    




            #     # Display the image using Matplotlib
            #     plt.imshow(grayscale_array, cmap='gray')
            #     plt.title("Transformed 28x28 Image")
            #     plt.show()


    if pygame.mouse.get_pressed()[0]:  # [0] corresponds to the left mouse button
        pygame.draw.circle(screen, white, (xcord, ycord), size_ratio)
        screen_array = pygame.surfarray.array3d(screen)

            #     # Transpose the array to (height, width, 3)
        screen_array = screen_array.transpose((1, 0, 2))

        # Convert to grayscale by averaging the RGB values
        grayscale_array = screen_array.mean(axis=2)

        # Normalize the grayscale array to [0, 1]
        grayscale_array = grayscale_array / 255.0

        # Convert to float32 to match model input type
        grayscale_array = grayscale_array.astype(np.float32)
        
        pred_lbl = model_predict(grayscale_array)





    text_surface = my_font.render(f"{pred_lbl}", True, blue,black)
    screen.blit(text_surface, (10, 10))




    fpsClock.tick(60)
    pygame.display.update()