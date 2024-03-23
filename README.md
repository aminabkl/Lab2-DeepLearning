# Amina BAKKALI TAHIRI

This lab focuses on exploring different deep learning architectures for image classification tasks using the MNIST dataset. It is divided into two main parts:

## Part 1: CNN Classifier

In this part, we implement Convolutional Neural Network (CNN) and Faster R-CNN models for classifying the MNIST dataset.

### Instructions:

1. **CNN Architecture:** Establish a CNN architecture using the PyTorch library. Define layers such as Convolution, Pooling, and Fully Connected layers along with hyper-parameters like kernels, padding, stride, optimizers, and regularization. Ensure the model is configured to run efficiently on GPU.
2. **Faster R-CNN:** Implement the Faster R-CNN architecture for the MNIST dataset.
3. **Model Comparison:** Compare the performance of the CNN and Faster R-CNN models using various metrics including accuracy, F1 score, loss, and training time.
4. **Fine-tuning with VGG16 and AlexNet:** Retrain the pre-trained VGG16 and AlexNet models on the MNIST dataset. Compare the results obtained with the CNN and Faster R-CNN models and draw conclusions.

## Part 2: Vision Transformer (VIT)

This part focuses on understanding Vision Transformers (ViT) and their application in image classification tasks.

### Instructions:

1. **VIT Model Architecture:** Follow the provided tutorial to establish a Vision Transformer (VIT) model architecture from scratch. Perform the classification task on the MNIST dataset.
2. **Result Interpretation and Comparison:** Analyze the obtained results and compare them with the outcomes from Part 1. Draw insights into the performance differences between CNN-based models and Vision Transformers.

## References:

- MNIST Dataset: [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- Tutorial for Vision Transformers: [Vision Transformers from Scratch - PyTorch](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
