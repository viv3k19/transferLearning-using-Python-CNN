# Transfer Learning in Image Classification

Transfer learning is a technique in machine learning where a pre-trained model, which has already learned features and patterns from a large dataset, is adapted to solve a new problem. This approach allows you to leverage the knowledge gained by the pre-trained model and apply it to your specific task, often with less data and computational resources than required when training a model from scratch.

## How Transfer Learning Works

In the context of deep learning, transfer learning often involves using a pre-trained neural network as a feature extractor for a new task. The lower layers of the network, which have learned general features from the source dataset, are kept, while the top layers, which are more task-specific, are replaced by new layers tailored to the target task. This process is known as fine-tuning. By freezing the lower layers and only training the new layers on the target dataset, the model can adapt to the new problem while still benefiting from the knowledge acquired on the source dataset.

## About this Repository

This repository contains a Jupyter Notebook (transferLearning.ipynb) that demonstrates the application of transfer learning in image classification. The notebook utilizes a pre-trained model from Google's TensorFlow Hub and retrains it on the flowers dataset. The goal is to showcase how transfer learning can be utilized to build an accurate image classification model with reduced training time and resources.

## Prerequisites

Before running the notebook, make sure you have the following prerequisites:

- Python 3.10 and ^
- Jupyter Notebook or JupyterLab installed
- Internet connection to download the required dependencies, pre-trained model, and dataset

## Technologies Used

The notebook utilizes the following technologies:

- Python: A widely used programming language for machine learning and data analysis.
- TensorFlow: An open-source machine learning framework for building deep learning models.
- TensorFlow Hub: A repository of pre-trained models that can be used for transfer learning.
- Pillow: A Python imaging library used for image manipulation and processing.
- NumPy: A library for numerical computations in Python.
- OpenCV: An open-source computer vision library used for image processing.
- Matplotlib: A plotting library for creating visualizations in Python.
- Jupyter Notebook: An interactive development environment for running Python code and creating documents.

## Dataset

The dataset used in this example is the [Flowers dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz). It contains images of 5 types of flowers:

- Roses
- Daisy
- Dandelion
- Sunflowers
- Tulips

## Pre-trained model

The pre-trained model used in this example is the [MobileNet V2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4) model from TensorFlow Hub.

## Workflow

1. Import necessary libraries and load the pre-trained MobileNet V2 model.
2. Make predictions using the pre-trained model (without any training).
3. Load the flowers dataset.
4. Read flower images from disk into a numpy array using OpenCV.
5. Split the dataset into training and testing sets.
6. Preprocess the images by scaling them.
7. Make predictions using the pre-trained model on the new flowers dataset.
8. Retrain the pre-trained model using the flowers dataset.
9. Evaluate the retrained model on the test set.

## How to Run the Notebook

To run the notebook, follow these steps:

1. Install the required dependencies: Pillow, TensorFlow Hub, NumPy, OpenCV, and Matplotlib.
2. Download the notebook (transferLearning.ipynb) and the required dataset from the provided links.
3. Launch Jupyter Notebook or JupyterLab in your local environment or preferred cloud platform.
4. Upload the notebook and the dataset to the Jupyter environment.
5. Open the notebook and execute the code cells sequentially.
6. Ensure that you have an internet connection to download the pre-trained model and the required dataset.

## Results and Evaluation

The notebook provides various insights and evaluations throughout the process. It includes making predictions using the pre-trained model, reading and preprocessing the flowers dataset, training and evaluating the retrained model, and making predictions on the new flowers dataset.

The model's performance can be evaluated using metrics such as accuracy, loss, and confusion matrix. The notebook includes visualization techniques, such as displaying sample images from the dataset and the predicted labels.

- Model Summary

![modelSummary](https://github.com/viv3k19/transferLearning-using-Python-CNN/assets/82309435/40ea0e4e-b943-484c-939e-3c0342756611)

- Trained for Five Epoch Only and got 92% Accuracy

![epochOnlyFive](https://github.com/viv3k19/transferLearning-using-Python-CNN/assets/82309435/16ff54fe-419a-4316-9704-685c802b81f6)


## Conclusion

Transfer learning is a powerful technique in machine learning that allows you to apply pre-trained models to new tasks and achieve accurate results with less training time and resources. By leveraging the knowledge and learned features of a pre-trained model, you can quickly adapt it to your specific problem domain. This notebook serves as a practical example of using transfer learning for image classification tasks, demonstrating the effectiveness and efficiency of the approach.

Please refer to the notebook (transferLearning.ipynb) for detailed code, explanations, and results.

# Project Creator
* Vivek Malam - Feel free to contact me at viv3k.19@gmail.com for any questions or feedback.

