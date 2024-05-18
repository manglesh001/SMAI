##Report Assignment_A4_2023201059
Which are different model i have used but does not give best score
CNN
Resnet18  with Hyperparameter  Tunning (weight decay=0.001,momentum=0.8,0.9, epoch =(20,25,30)
Resnet34   with Hyperparameter Tunning (weight decay=0.001,momentum=0.8,0.9, epoch =(20,25,30)
Resnet50   with Hyperparameter Tunning (weight decay=0.001,momentum=0.8,0.9, epoch =(20,25,30)
Resnet101  with Hyperparameter Tunning (weight decay=0.001,momentum=0.8,0.9, epoch =(20,25,30)
efficientnet_b0 with Hyperparameter Tunning epoch =(20,30)
efficientnet_b1 with Hyperparameter Tunning  epoch =(20,25,30)


#Best Model
Pretrained Model=EfficientNet b0
Epoch=25
learning rate=0.001
optimizer=Adam
Loss=L1 loss

Age Prediction using the EfficientNet b0  with 25 epoch architecture in PyTorch. 

Imports necessary libraries 

AgeDataset Class: This class defines a custom dataset for loading images and their corresponding ages. It reads image filenames and ages from CSV files and performs transformations such as resizing, converting to RGB, and normalization.

AgePredictionModel Class:   EfficientNet backbone followed by a fully connected layer with one neuron for age prediction.

Training the Model: iterates through the dataset for a specified number of epochs 25 , computes the  MAE L1 loss, and updates the model parameters using the Adam optimizer.

Prediction Function:  make predictions using the trained model. It iterates through the test dataset, makes predictions, and stores them in a list.

Loading Data and Model: It loads the training and test datasets using the AgeDataset class and creates data loaders for both.

Training the Model: train_model function to train the model using the training data(train.csv).

Making Predictions:predict function to make predictions on the test data (test.csv).

Saves the predictions along with image IDs into a CSV file(2023201059_A4.csv).



