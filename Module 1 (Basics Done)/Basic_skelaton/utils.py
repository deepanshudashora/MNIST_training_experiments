import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# Data to plot accuracy and loss graphs

import os
os.makedirs("logs/", exist_ok=True)

import logging
logging.basicConfig(filename='logs/network.log', format='%(asctime)s: %(filename)s: %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)



test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def get_device():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logger.info("device: %s" % device)
    return device


def data_transformation(transformation_matrix):
    # Train Transform
    logger.info("transformation Details ::: ", transformation_matrix)
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(transformation_matrix["mean_of_data"],transformation_matrix["std_of_data"])
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(transformation_matrix["mean_of_data"],transformation_matrix["std_of_data"])
        ])

    return train_transforms, test_transforms


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer,train_losses,train_acc):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = F.nll_loss(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
  return train_losses,train_acc

def test(model, device, test_loader,test_losses,test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_losses,test_acc


def fit_model(model,training_parameters,train_loader,test_loader,device):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    optimizer = optim.SGD(model.parameters(), lr=training_parameters["learning_rate"], momentum=training_parameters["momentum"])
    for epoch in range(1, training_parameters["num_epochs"]+1):
        print(f'Epoch {epoch}')
        train_losses,train_acc = train(model, device, train_loader, optimizer,train_losses,train_acc)
        test_losses,test_acc = test(model, device, test_loader,test_losses,test_acc)  
    logging.info('Training Losses : %s', train_losses)
    logging.info('Training Acccuracy : %s', train_acc)
    logging.info('Test Losses : %s', test_losses)
    logging.info('Test Accuracy : %s', test_acc)
        
    return train_losses, test_losses, train_acc, test_acc

def plot_accuracy_report(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def show_random_results(test_loader,grid_size,model,device):
  cols, rows = grid_size[0],grid_size[1]
  figure = plt.figure(figsize=(20, 20))
  for i in range(1, cols * rows + 1):
      k = np.random.randint(0, len(test_loader.dataset)) # random points from test dataset
    
      img, label = test_loader.dataset[k] # separate the image and label
      img = img.unsqueeze(0) # adding one dimention
      pred=  model(img.to(device)) # Prediction 

      figure.add_subplot(rows, cols, i) # adding sub plot
      plt.title(f"Predcited label {pred.argmax().item()}\n True Label: {label}") # title of plot
      plt.axis("off") # hiding the axis
      plt.imshow(img.squeeze(), cmap="gray") # showing the plot

  plt.show()

def plot_misclassified(model,grid_size,test_loader,device):
  count = 0
  k = 0
  misclf = list()
  while count<=20:
    img, label = test_loader.dataset[k]
    pred = model(img.unsqueeze(0).to(device)) # Prediction
    pred = pred.argmax().item()

    k += 1
    if pred!=label:
      misclf.append((img, label, pred))
      count += 1
  
  rows, cols = grid_size[0],grid_size[1]
  figure = plt.figure(figsize=(20,20))

  for i in range(1, cols * rows + 1):
    img, label, pred = misclf[i-1]

    figure.add_subplot(rows, cols, i) # adding sub plot
    plt.title(f"Predcited label {pred}\n True Label: {label}") # title of plot
    plt.axis("off") # hiding the axis
    plt.imshow(img.squeeze(), cmap="gray") # showing the plot

  plt.show()

# For calculating accuracy per class
def calculate_accuracy_per_class(model,device,test_loader,test_data):  
  model = model.to(device)
  model.eval()
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = model(images.to(device))
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels).squeeze()
          for i in range(10):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1
  final_output = {}
  classes = test_data.classes
  for i in range(10):
      print('Accuracy of %5s : %2d %%' % (
          classes[i], 100 * class_correct[i] / class_total[i]))
      final_output[classes[i].split("-")[1]] = 100 * class_correct[i] / class_total[i]
      
  original_class = list(final_output.keys())
  class_accuracy = list(final_output.values())
  plt.figure(figsize=(8, 6))
  plt.bar(original_class, class_accuracy)
  plt.xlabel('classes')
  plt.ylabel('accuracy')
  plt.show()