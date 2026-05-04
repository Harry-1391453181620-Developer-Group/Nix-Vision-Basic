# Nix-Vision Basic

A fully **from-scratch Convolutional Neural Network (CNN)** built using
only **NumPy**, designed to classify images into multiple categories.

------------------------------------------------------------------------

## What does this AI do?

This model performs **image classification** on a **13-class dataset**,
predicting which category an input image belongs to.

### Classes:

airplane, bird, cat, deer, dog, frog, horse, man, maple_tree,
motorcycle, mountain, mouse, mushroom

------------------------------------------------------------------------

## Model Architecture

Input (1×64×64)\
→ Conv (32) + ReLU + MaxPool\
→ Conv (64) + ReLU + MaxPool\
→ Global Average Pooling\
→ Fully Connected (64)\
→ Dropout\
→ Fully Connected (num_classes)\
→ Softmax

------------------------------------------------------------------------

## Current Progress

-   Training pipeline fully working\
-   Stable convergence\
-   Validation accuracy \~45--50%\
-   Currently improving model capacity

------------------------------------------------------------------------

## Training

Run the command stored in:

train_command.txt

### In PowerShell:

python train.py @args

------------------------------------------------------------------------

## Saved Model

The trained model is saved at:

root/model.npz

------------------------------------------------------------------------

## Prediction

Use:

predict.py

------------------------------------------------------------------------

## Dependencies

-   Python \>= 3.12.3\
-   numpy\
-   Pillow (PIL)\
-   A 13-class dataset

------------------------------------------------------------------------