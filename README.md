# 👟 neural-net-fashion-mnist

A beginner-friendly AI project that implements a neural network model to classify images of clothing items from the Fashion MNIST dataset using TensorFlow and Keras. This dataset is a well-known benchmark in machine learning and computer vision, featuring 70,000 grayscale images (28x28 pixels) across 10 classes like shirts, dresses, sneakers, and more. The project walks through a complete AI pipeline — from loading and preprocessing data to training a model and visualizing predictions. It's beginner-friendly and ideal for students and early learners exploring neural networks.

Key steps include:
Preprocessing: Input images are normalized (values scaled to 0–1) and flattened from 28x28 matrices into 784-element vectors for the dense neural network.

Model Architecture: A Sequential neural network with two hidden layers (128 and 64 units, using ReLU activation) and an output layer with 10 units (softmax activation) for multi-class classification.

Training: The model is trained for 10 epochs using the Adam optimizer and sparse categorical crossentropy loss. A validation split ensures the model generalizes well.

Evaluation & Visualization: After training, model accuracy is evaluated on the test dataset, and sample predictions are visualized alongside actual labels. Class numbers are converted to human-readable names (e.g., “Ankle boot” instead of “9”).

The final model achieves ~85% accuracy on test data, and predictions are visualized using matplotlib.

This project is excellent for building intuition around dense neural networks before moving on to convolutional models (CNNs). Future improvements could include using dropout layers, adding CNNs for higher accuracy, or experimenting with color image datasets like CIFAR-10.

## 📌 Overview

- **Dataset**: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- **Model**: Dense Neural Network (DNN)
- **Framework**: TensorFlow 2.x + Keras
- **Goal**: Classify grayscale 28x28 images into 10 fashion categories
- **Test Accuracy**: ~85%

## 🧾 Class Labels

0 - T-shirt/top
1 - Trouser
2 - Pullover
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle boot
