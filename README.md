## Dog Breed Image Classifiation using Sagemaker

In this project, I used AWS Sagemaker to finetune a pretrained model(ResNet34) in order to perform image classification on different Breeds of Dogs.

# Image Classification using AWS SageMaker

AWS Sagemaker was used to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 
This can be done on either the provided dog breed classication data set or one of your choice.
The jupyter notebook "train_and_deploy.ipynb" walks you through the implementation of Image Classification Machine Learning Model to classify between 133 kinds of dog breeds using dog breed dataset provided by Udacity Link - (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
The concept of transfer learning would be implored in this project using a ResNet34 pretrained model alongside three Fully connected neural network layer. We would then perform Hyperparameter tuning to help figure out the best hyperparameters to be used for our model.
Thereafter, we would make use of the best hyperparameters and fine-tuning our Resent34 model.
After which we would be making use of the Profiling and Debugging configuration  in our training mode by adding in relevant hooks in the Training and Testing( Evaluation) phases.
Next we will be deploying our model. While deploying we will create our custom inference script. The custom inference script will be overriding a few functions that will be used by our deployed endpoint for making inferences/predictions.
Finally we will be testing out our model with some test images of dogs, to verfiy if the model is working as per our expectations.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available.
Download dependencies and modules


## Dataset
The Dataset used was gotten from the udacity classroom.it contains about a thousand images of dogs made up of a total of 133 different breeds.
The dataest consist of the train, test and validation data


### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data using the aws s3 sync bucket command.. 
(s3_data.png)

## Hyperparameter Tuning

The pretrained model used was ResNet34 model with a three Fully connected Linear NN layer's in order to flatten the result befroe classifying it is used for this image classification problem. https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html )
Hence, the hyperparameters selected for tuning were:
Learning rate - default(x) is 0.001 , so  I have selected 0.01x to 100x range for the learing rate
Batch size -- selected only== two values [ 32, 64, 128,256 ]
Best Hyperparamters was :
{'batch_size': '"64"','lr': '0.0038180486809046526'}
(training_jobs.PNG)
(best_hyperparameter_tuning_job.PNG)


## Debugging and Profiling
The Debugger Hook is set to record the Loss Criterion of the process in both training and validation/testing. The Plot of the Cross Entropy Loss is shown below.
!(cross_entropy_loss.PNG)

### Results
From the cross_entropy_loss graph, the line is not smooth and with high and low spikes during the validation phase.
Making some adjustments in the pretrained model to use a different set of the fully connected layers network,
This should help to smoothen out the graph also changing the weights and either by adding more fully connected layers or 
using only one fully connected layer might smoothen the plots depending.


## Model Deployment.
in deploying the model, I added a new inference script named 'endpoint2.py' and deplloyed the model on the ml.g4dn.xlarge instance.
 The inference from the test_data was about   100% Accurate. of the 4 test_data, the model predicted 4 out of 4 correctly.
Below is the scrrenshot of the endpoint deployed 

