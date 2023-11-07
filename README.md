# Convolutional Neural Network (CNN) Project

This project is focused on learning and implementing Convolutional Neural Networks (CNNs) using PyTorch for image classification. I have used the MNIST dataset for this task, which consists of hand-written digits. The project is divided into two files, each representing a different stage of the CNN implementation.

## File 1(CNN_Model_Breakdown): Exploring CNN Basics

- In "CNN_Model_Breakdown.ipynb," I have created the breakdown of all the fundamental components of a CNN. The following key steps are performed:

### Step 1: Data Preparation

- Used PyTorch and the torchvision library to download and prepare the MNIST dataset for training and testing. The data is converted into tensors.

### Step 2: Data Loading

- Created DataLoader objects for both the training and test datasets, enabling us to efficiently load batches of data for training and evaluation.

### Step 3: Defining the CNN Model

- Defined a simple CNN model with two convolutional layers (`conv1` and `conv2`) to learn feature representations from the input images.

### Step 4: Understanding Convolution and Pooling

- Explained the concept of convolution and pooling in CNNs, demonstrating their impact on the dimensions of the feature maps.

### Step 5: Processing a Single Image

- Performed the initial processing of a single MNIST image using the defined model, showing the changes in the image dimensions after each layer.

## File 2(CNN_Model): Additional Experiments

- In "CNN_Model.ipynb," I have created a model for exploring CNN concepts and demonstrate additional experiments or code snippets.

### Model Structure

- Created a sample Model class with 2 convolutional layer along with maxpooling layers 3 Fully-connected layers. And then feed the training data to the model and implemented an end to end CNN model.

### Visualizing Data

- Vizualized the data or display intermediate results to enhance understanding.

## Prerequisites

- To run these scripts, you should have Python installed along with the required libraries, including PyTorch, NumPy, and Matplotlib.

## Getting Started

- To get started, clone this repository and execute the Jupyter notebooks "CNN_Model_Breakdown.ipynb" and "CNN_Model.ipynb." Make sure you have the necessary dependencies installed.

```bash
git clone https://github.com/your-username/cnn-project.git
cd CNN-MODEL
jupyter notebook file1.ipynb
```