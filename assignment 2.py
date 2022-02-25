import os
import sys
import time
from torch.utils.data import Dataset
import scipy.io as scio
from skimage import io
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.optim import Adam, SGD
import itertools
import matplotlib.image as mpimg
import matplotlib
import warnings


#function to get rid of warning messages
warnings.filterwarnings('ignore')


#Task 1
#REWRITTEN TASK 1 AND 2 TO USE FORWARD
class Convolutional_Neural_Network(nn.Module):

    def __init__(self):
        super(Convolutional_Neural_Network, self).__init__()

        #setting up the layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        batch_layer = nn.BatchNorm2d(64)
        self.relu = lambda x : nn.functional.relu(batch_layer(x))

        self.fc = nn.Linear(57600, 5)


    #now using forward function to actually step through the layers as it wasnt before
    def forward(self, x):
        # First hidden layer
        x = self.conv1(x)
        x = self.relu(x)

        # Second hidden layer
        x = self.pooling(x)

        # Third hidden layer
        x = self.conv2(x)
        x = self.relu(x)

        # Fourth hidden layer
        x = self.pooling(x)

        # Fully connected layer, with the output channel 5
        x = x.view(-1, 64 * 30 * 30)
        x = self.fc(x)
        
        return x


#Task 2
#learning rates are defined within the epochs_and_rates class
def loss_and_optimiser(cnn, learning_rate=0.001):
    cross_loss = CrossEntropyLoss()
    optimiser = Adam(cnn.parameters(), lr=learning_rate)

    return cross_loss, optimiser


#define and print the model
cnn = Convolutional_Neural_Network()
print(cnn)


##LOADING IMAGES USING THE GIVEN imgdata.py file
class imageDataset(Dataset): 

    def __init__(self, root_dir, file_path, imSize = 250, shuffle=False):
        self.imPath = np.load(file_path) 
        self.root_dir = root_dir
        self.imSize = imSize
        self.file_path=file_path


    def __len__(self):
        return len(self.imPath)

    def __getitem__(self, idx):
    	# print(self.root_dir)
    	# print(self.imPath[idx])
        im = io.imread(os.path.join(self.root_dir, self.imPath[idx]))  # read the image

        if len(im.shape) < 3: # if there is grey scale image, expand to r,g,b 3 channels
            im = np.expand_dims(im, axis=-1)
            im = np.repeat(im,3,axis = 2)

        img_folder = self.imPath[idx].split('/')[-2]
        if img_folder =='faces':
            label = np.zeros((1, 1), dtype=int)
        elif img_folder == 'dog':
            label = np.zeros((1, 1), dtype=int)+1
        elif img_folder == 'airplanes':
            label = np.zeros((1, 1), dtype=int)+2
        elif img_folder == 'keyboard':
            label = np.zeros((1, 1), dtype=int)+3
        elif img_folder == 'cars':
            label = np.zeros((1, 1), dtype=int)+4


        img = np.zeros([3,im.shape[0],im.shape[1]]) # reshape the image from HxWx3 to 3xHxW
        img[0,:,:] = im[:,:,0]
        img[1,:,:] = im[:,:,1]
        img[2,:,:] = im[:,:,2]

        imNorm = np.zeros([3,im.shape[0],im.shape[1]]) # normalize the image
        imNorm[0, :, :] = (img[0,:,:] - np.max(img[0,:,:]))/(np.max(img[0,:,:])-np.min(img[0,:,:])) -0.5
        imNorm[1, :, :] = (img[1,:,:] - np.max(img[1,:,:]))/(np.max(img[1,:,:])-np.min(img[1,:,:])) -0.5
        imNorm[2, :, :] = (img[2,:,:] - np.max(img[2,:,:]))/(np.max(img[2,:,:])-np.min(img[2,:,:])) -0.5

        return{
            'imNorm': imNorm.astype(np.float32),
            'label':np.transpose(label.astype(np.float32))                  #image label
            }


class DefaultTrainSet(imageDataset):

    def __init__(self, **kwargs):
        script_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
         #  img_list_train.npy that contains the path of the training images is provided 
        default_path = os.path.join(script_folder, 'img_list_train.npy')
        root_dir = os.path.join(script_folder, 'data')
        super(DefaultTrainSet, self).__init__(root_dir, file_path=default_path, imSize = 250,**kwargs)


class DefaultTestSet(imageDataset):

    def __init__(self, **kwargs):
        script_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        #  img_list_test.npy that contains the path of the testing images is provided
        default_path = os.path.join(script_folder, 'img_list_test.npy')  
        root_dir = os.path.join(script_folder, 'data')
        super(DefaultTestSet, self).__init__(root_dir, file_path=default_path, imSize = 250,**kwargs)


class epochs_and_rates(object):
    #lazy check if cpu is available
    dev_available = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #set our batch and epoch size
    batch_size = 16
    num_epochs = [5, 20]
    #set our learning rates to 0.01, 0.001, 0.0001, 0.00001
    learning_rates = [1e-02, 1e-03, 1e-04, 1e-05]

    #load the training and testing images
    training = imageDataset('data', 'img_list_train.npy')
    testing = imageDataset('data', 'img_list_test.npy')

    #calculate the length of training set and testing set for iteration and testing
    num_train = len(training)
    num_test = len(testing)


def load_model():
    #create a dictionary of trained models
    trained_models = {}
    #for each model, loop by lr
    for num_epochs in epochs_and_rates.num_epochs:
        for rate in epochs_and_rates.learning_rates:
            #load the cnn
            cnn = Convolutional_Neural_Network()
            cnn.load_state_dict(torch.load(generate_path(num_epochs, rate)))
            trained_models[rate] = cnn
    return trained_models


#generate the path for each model
def generate_path(num_epochs, learning_rate):
    return f"models/model_epochs-{num_epochs}_learning_rate-" + "{:.0e}".format(learning_rate) + ".pth"


def train(cnn, batch_size, num_epochs, learning_rate):
    cross_loss, optimiser = loss_and_optimiser(cnn, learning_rate)
    minibatch = epochs_and_rates.num_train // batch_size
    #plot vals
    #train hist will contain the loss and accuracy of each epoch
    train_hist = []
    acc_hist = []
    
    for epoch in range(num_epochs):
        train_loss = 0
        accurate_classifications = 0

        for mini in range(minibatch):
            #mini_x is the data, mini_y is the labels
            mini_x = torch.tensor([epochs_and_rates.training[j]['imNorm'] for j in range(mini, mini+batch_size)], dtype=torch.float32)
            mini_y = torch.tensor([epochs_and_rates.training[j]['label'] for j in range(mini, mini+batch_size)], dtype=torch.int64)
            #reshaping the numpy array from a 3d to a 1d array, as we only need the labels
            x,y,z = mini_y.shape
            d1_mini_y = mini_y.reshape((x))
            
            #set the parameter gradients to 0 as this is needed for updating the weights and biases
            #for each minibatch
            #needed for backpropragation
            optimiser.zero_grad()

            #forward and backward propragation
            out = cnn(mini_x)
            loss = cross_loss(out, d1_mini_y)

            #storing the accuracies for each class, selecting the highest one to make prediction
            accurate_classifications += torch.sum(torch.argmax(out, dim = 1) == d1_mini_y)
            train_loss += loss.item()

            loss.backward()
            optimiser.step()

        #loading and saving the model
        model = generate_path(epoch+1, learning_rate)
        torch.save(cnn.state_dict(), model)

        #calculating the average loss per epoch
        avg_loss = train_loss / epochs_and_rates.num_train
        train_hist.append(avg_loss)
        
        #calculating the accuracy per epoch
        accuracy = accurate_classifications / epochs_and_rates.num_train
        acc_hist.append(accuracy)

    #load the model and return the training history and accuracy
    cnn.load_state_dict(torch.load(model))
    return train_hist, acc_hist


def plotting(train_hist):
    fig, (y1, y2) = plt.subplots(1, 2)
    y1.set_title('Training Loss.')
    y2.set_title('Training Accuracy.')
    plt.xlim = ([0,20])
    plt.ylim = ([0,20])
    for rate, (loss_hist, acc_hist) in train_hist.items():
        x = np.arange(1, len(loss_hist) + 1)
        y1.plot(x, loss_hist, label=rate)
        y2.plot(x, acc_hist, label=rate)

    y1.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))
    y2.xaxis.set_ticks(np.arange(min(x), max(x)+1, 1.0))

    handlers, labels = y2.get_legend_handles_labels()
    plt.legend(handlers, labels, loc='lower right')

    plt.setp(y1, xlabel='Epoch', ylabel='Loss')
    plt.setp(y2, xlabel='Epoch', ylabel='Accuracy')
    plt.show()


def plot_confusion_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


def testingPhase(batch_size):

    trained_models = load_model()
    cross_loss, optimiser = loss_and_optimiser(trained_models[1e-04], 1e-04)

    minibatch = epochs_and_rates.num_test // batch_size

    test_loss = 0
    accurate_classifications = 0

    for mini in range(minibatch):
        #mini_x is the data, mini_y is the labels
        mini_x = torch.tensor([epochs_and_rates.testing[j]['imNorm'] for j in range(mini, mini+batch_size)], dtype=torch.float32)
        mini_y = torch.tensor([epochs_and_rates.testing[j]['label'] for j in range(mini, mini+batch_size)], dtype=torch.int64)
        #reshaping the numpy array from a 3d to a 1d array, as we only need the labels
        x,y,z = mini_y.shape
        d1_mini_y = mini_y.reshape((x))
        
        #forward and backward propragation
        out = cnn(mini_x)
        loss = cross_loss(out, d1_mini_y)

        #storing the accuracies for each class, selecting the highest one to make prediction
        accurate_classifications += torch.sum(torch.argmax(out, dim = 1) == d1_mini_y)
        test_loss += loss.item()

    total_loss = test_loss / epochs_and_rates.num_test
    total_accuracy = accurate_classifications / epochs_and_rates.num_test

    print("HELP")
    return


if  __name__ == "__main__":
    #if a model already exists then don't train it again
    #THIS SAVES A LOT OF TIME
    if not os.path.isfile('train_history_dictionary.npy'):
        #for each learning rate (of each epoch) save the training history
        train_hist = {}
        for rate in epochs_and_rates.learning_rates:
            train_hist[rate] = train(cnn, batch_size = 16, num_epochs = max(epochs_and_rates.num_epochs), learning_rate = rate)
        np.save('train_history_dictionary.npy', train_hist, allow_pickle = True)
    train_hist = np.load('train_history_dictionary.npy', allow_pickle = True)[()]
    plotting(train_hist)
    testingPhase(batch_size = 50)
