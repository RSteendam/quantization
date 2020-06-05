# quantization
This repository contains the code for the experiments ran by Ruben Steendam for his bachelor thesis *'Testing Mixed
-Precision for VGG, Inception and ResNet on the Dogs vs. Cats Dataset'*. This thesis was written for the bachelor
 Artificial Intelligence in the year 2020 and was supervised by Dr. Thijs van Ommen.
 
 The repository contains unpolished work and is not meant for day-to-day use.
 
 ## Initial setup
 1. Setup a TensorFlow capable machine with a mixed-precision GPU. GPU compute capability 7.0 or higher (https
 ://developer).
 .nvidia
 .com/cuda-gpus 
 2. Update the first-deploy.sh script with your own IP and ssh username.
 3. Update the ```utils/kaggle.json``` file with your own kaggle credentials and enroll in the dogs-vs-cats
  competition (https://www.kaggle.com/c/dogs-vs-cats/data).
 
 ## Running the experiments
 The experiments can be run using the ```run.py``` script. This script trains or tests networks on both single- and
  mixed-precision. 
  
 Here the following arguments can be used:
 * ```--batch_size```, this sets the batch size to either 16, 32, 64 or 128
 * ```--epochs```, set the number of epochs for the models to train
 * ```--model```, specify the model to run the experiments on, either vgg, resnet or inception
 * ```--run```, specify the type of experiments to run. train, test, training-speed or inference-speed. Train and
  training-speed can be run after the initial setup, test and inference-speed after training the corresponding model
  . Train trains the network and reports its accuracy (both training and validation), test returns the mcnemar test
   scores, training-speed the training speed for the full unfrozen network and inference-speed the inference speed. 
 