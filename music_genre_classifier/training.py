import torch
from torch import nn
from torchsummary import summary
import torch.optim as optim
from models.cnn import CNNNetwork
import data_managment.dataset as dataset
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import os

EPOCHS = 300
LEARNING_RATE = 0.001

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.dataloader.DataLoader, 
          valid_dataloader: torch.utils.data.dataloader.DataLoader, 
          epochs: int, 
          loss_function: torch.nn.modules.loss,
          optimiser: torch.optim, 
          device: str) -> tuple:
    '''Trains and validates a CNN model.
    
    Parameters
    ----------
    model : torch.nn.Module
        CNN model to be trained. 
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader. 
    valid_dataloader : torch.utils.data.DataLoader
        Validation dataloader.
    epochs : int > 0
        Number of epochs.
    loss_function : torch.nn.modules.loss
        Loss function. 
    optimiser : torch.optim
        Optimiser.
    device : str
        Device to process. CPU or GPU (cuda). 
        
    Returns
    -------
    model : torch.nn.Module
        Model trained to be saved.
    history : dict
        Dictionary with training and validation information (loss and accuracy).
    '''
    history = {
        'training_loss': [],
        'training_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
        }
    
    for epoch in range(epochs):
        
        # Training
        model, train_loss, train_accuracy = train_single_epoch(model, train_dataloader, loss_function, optimiser, device)
        history['training_loss'].append(train_loss), history['training_accuracy'].append(train_accuracy)
        
        # Validation
        val_loss, val_accuracy = validate_single_epoch(model, valid_dataloader, loss_function, optimiser, device)
        history['validation_loss'].append(val_loss), history['validation_accuracy'].append(val_accuracy)
        
        print(f'Epoch {epoch+1} | {epochs} --> Train loss: {train_loss:.4f}      | Train accuracy: {train_accuracy:.4f}')
        print(f'Epoch {epoch+1} | {epochs} --> Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.4f}')
        print('-------------------------------------------------------------------------')
    
    return model, history

def train_single_epoch(model: torch.nn.Module, 
                       train_dataloader: torch.utils.data.dataloader.DataLoader, 
                       loss_function: torch.nn.modules.loss,
                       optimiser: torch.optim, 
                       device: str) -> tuple:
    '''Trains a single epoch of a model's training. 

    Parameters
    ----------
    model : torch.nn.Module
        CNN model to be trained. 
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader. 
    loss_function : torch.nn.modules.loss
        Loss function. 
    optimiser : torch.optim
        Optimiser.
    device : str
        Device to process. CPU or GPU (cuda). 
        
    Returns
    -------
    model : torch.nn.Module
        Model trained in a single epoch.
    train_loss: float
        Training loss in a single epoch.
    train_accuracy: float
        Training accuracy in a single epoch.
    '''
    train_loss = []
    y_true = []
    y_pred = []
    model.train()
    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)
        
        prediction = cnn(data) # Predict data with the model
        loss = loss_function(prediction, labels) # Loss between predicted values and label values
        optimiser.zero_grad() # Zero out the gradients
        loss.backward() # Backpropagation. Computes gradients
        optimiser.step() # Updates the weights
        train_loss.append(loss.item())
        
        # Accuracy
        _, pred = torch.max(prediction.data, 1) # Prediction with most probability
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist()) 
        
    train_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    train_loss = np.mean(train_loss)
    
    return model, train_loss, train_accuracy
    
def validate_single_epoch(model: torch.nn.Module, 
                          valid_dataloader: torch.utils.data.dataloader.DataLoader, 
                          loss_function: torch.nn.modules.loss,
                          optimiser: torch.optim, 
                          device: str) -> tuple:
    '''Validates a single epoch of a model's training. 
    
    Parameters
    ----------
    model : torch.nn.Module
        CNN model to be trained. 
    valid_dataloader : torch.utils.data.DataLoader
        Validation dataloader.
    loss_function : torch.nn.modules.loss
        Loss function. 
    optimiser : torch.optim
        Optimiser.
    device : str
        Device to process. CPU or GPU (cuda). 
    
    Returns
    -------
    valid_loss: float
        Validation loss in a single epoch.
    train_accuracy: float
        Validation accuracy in a single epoch.
    '''
    valid_loss = []
    y_true = []
    y_pred = []
    model.eval()
    for data, labels in valid_dataloader:
        data, labels = data.to(device), labels.to(device)
           
        prediction = model(data) # Predict data with the model
        loss = loss_function(prediction, labels) # Loss between predicted values and label values
        valid_loss.append(loss.item())
    
        # Accuracy
        _, pred = torch.max(prediction.data, 1) # Prediction with most probability
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist()) 
        
    val_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    valid_loss = np.mean(valid_loss)
    
    return valid_loss, val_accuracy

def plot_loss_accuracy(history: dict, epochs: int):
    '''Plot the accuracy and the loss of a model on training and validation. 

    Parameters
    ----------
    history : dict
        Model's history. Accuracy and loss information on training and validation.
    epochs : int > 0
        Number of epochs. 
        
    Returns
    -------
    None.
    '''
    plt.subplot(211)
    plt.plot(range(1, epochs+1), history['training_loss'], 'bo', label='Training loss')
    plt.plot(range(1, epochs+1), history['validation_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.show()
    plt.subplot(212)
    plt.plot(range(1, epochs+1), history['training_accuracy'], 'bo', label='Training accuracy')
    plt.plot(range(1, epochs+1), history['validation_accuracy'], 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().set_yticks(plt.gca().get_yticks())
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
    plt.legend()
    plt.grid('off')
    plt.show()

if __name__ == '__main__':
    
    # Dataset
    train_dataloader = dataset.train_loader
    valid_dataloader = dataset.valid_loader
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    # Construct a model
    cnn = CNNNetwork().to(device)
    summary(cnn, (1, 128, 130)) 
    
    # Initialise Loss function and Optimizer
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Train model
    model, history = train(model=cnn,
                           train_dataloader=train_dataloader, 
                           valid_dataloader=valid_dataloader, 
                           epochs=EPOCHS,
                           loss_function=loss_function,
                           optimiser=optimiser,
                           device=device)

    # Save model
    if os.path.exists('results'):
        pass
    else:
        os.makedirs('results') 
    torch.save(model.state_dict(), 'results/music_genre_classifier.pth')
       
    # Plot history
    plot_loss_accuracy(history=history, epochs=EPOCHS)
    
