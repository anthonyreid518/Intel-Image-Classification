# Intel Image Classification 

## Problem Statement
How can we build a machine learning model to classify images as either buildings, forest, glacier, mountains, streets, or the sea?

## Executive Summary
Through this project, images of buildings, forest, glacier, mountains, streets, and the sea were gathered and used to build a neural net classification to classify the images. The dataset was provided through Kaggle had roughly 14,000+ images for training and 3,000+ images for training. Accuracy was used as the metric for monitoring the performance of the models, and the baseline model (simple neural net with 1 hidden layer) had an accuracy of 38%. This was the score to beat. Improvements on the baseline model included adding more hidden layers, dropout layers, early stopping, kernel regularization, convolution layers, and pooling layers. My reccomendation is to use a convolution neural net with 2 convolution layers, max pooling layers, and learning rate reduction to reduce overfitting. This model has a final accuracy score of 76% surpassing our base model.

## Table of Content
|File Name|Description|
|---|---|
|**Seg_pred**|Folder with 7,302 images to apply to the model|
|**train**|Folder with 6 sub folders (buildings, forest, street, mountain, sea, glacier) to train on. 14,041 images in total|
|**train_buildings**|Subfolder of train with 2,192 images of buildings|
|**train_forest**|Subfolder of train with 2,272 images of forests|
|**train_street**|Subfolder of train with 2,383 images of streets|
|**train_mountain**|Subfolder of train with 2,513 images of mountains|
|**train_sea**|Subfolder of train with 2,275 images of the sea|
|**train_glacier**|Subfolder of train with 2,405 images of glaciers|
|**Val**|Folder with 6 sub folders (buildings, forest, street, mountain, sea, glacier) to validate on. 3,007 images in total. Similar format to train folder|
|**Best-Model.ipynb**|Python notebook file with the best convolution neural net model|
|**Classification-models.ipynb**|Python notebook containin all of the models tested|
|**satellite.py**|Python file used to run streamlit app|

## Data Accumilation
The data used in this project was provided through kaggle. The purpose was for an image classification challenge hosted by Intel. The link for the dataset is (https://www.kaggle.com/puneet6060/intel-image-classification)

## Conclusion and Future Work
Using a convolution neural net dramatically increases the perfomance of the model and helps with properly classifying the labels. In order to reduce overfitting, a learning rate reduction was applied to the model. Also to decrease runtime, earlystopping was added. Although the model performs well there may be room for improvement or further experimentation such as transfer learning and data augmentation. 