"""
This file contains template of a classical Deep Learning architecture (Feedforward Neural Network) inspired by the research of [cite] and designed by Rossen Boykov. It is used in variety of experiments in the coresponding git folder.

- Activations: ReLU
- Data: TMP
- Good graphs
- Option to import seed
- Multiple runs 
"""

### Imports
from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
#import nn.functional
import torch.nn.functional as F
import matplotlib
import sys
from torch.utils.tensorboard import SummaryWriter
import csv
import time

# Hyperparameters
input_size = 1
num_classes = 1
number_hidden = 128
learning_rate = 0.002
batch_size = 1
num_epochs = 9000
validation_size = 1100
a_param = 30
batch_size = 16

random_seed = torch.get_rng_state()
"""
random_seed = random_seed2
for i, elem in enumerate(random_seed):
  random_seed[i] = int(elem)
random_seed = torch.Tensor(random_seed).type(torch.uint8)
print('Test1: ', random_seed[:10])
torch.set_rng_state(random_seed)
"""


plt.rcParams['figure.figsize'] = [13, 6]
model_name = 'FNN-Snake-split'
study_name = 'Identity and peroidic component'
font = {'family': 'normal',
        'size': 12}
matplotlib.rc('font', **font)


### Data preparation ###

count = 0
acc_list = []
### Data preparation ###
#1) option for FNN original experiment
file1 = open('./produkt_klima_tag_19540601_20201231_03379.txt', 'r')
Lines = file1.readlines()


# Removing NaN values and converting to float


for i, line in enumerate(Lines):
    if i == 0:
      continue
    #print(line.strip().split(';')[3])
    value = np.float32(line.strip().split(';')[13])
    acc_list.append(value)

#print('L tmp: ', len(tmp), ' AccList: ', len(acc_list))
#acc_list = tmp
acc_list = acc_list[20500:]
print('Removed Row: ', count)
print('The data 1: ', acc_list)
print('Len of acc_list: ', len(acc_list))
acc_list = np.array(acc_list)


print('The data 2: ', acc_list)
plt.plot(acc_list, color='orange')  # acc_list[:6300]
plt.xlabel('day')
plt.ylabel('Temperature Measurements (extended)')
plt.savefig('./GroundTruth.png')
plt.show(block=False)

# Snake


def Snake(x):
  a = a_param
  return x + torch.sin(a*x)*torch.sin(a*x)/a


def Snake_split2(x):
  a = a_param
  #print('The Mr X: ', len(x), len(x)//2, type(x))
  x_tmp = torch.sin(a*x)*torch.sin(a*x)/a
  x_ind = x

  #print('The Per: ', type(x_tmp), x_tmp.size())
  #print('The IND: ', type(x_ind), x_ind.size())
  x_ret = torch.cat( (x_tmp[1::2], x_ind[0::2] ))
  return x_ret
# Model

def Snake_i(x):
  a = a_param
  return x


def Snake_p(x):
  a = a_param
  return torch.sin(a*x)*torch.sin(a*x)/a


def Snake_splitter(self, x = None):
    if x == None:
        return None
    linear_out = x
    split_point = len(x)/2
    print('X sizer: ', x.size())
    first_slice = linear_out[:, 0:split_point]
    second_slice = linear_out[:, split_point:]
    tuple_of_activated_parts = (
        Snake_i(first_slice),
        Snake_p(second_slice)
    )
    # concatenate over feature dimension
    return torch.cat(tuple_of_activated_parts, dim=1)



class Net(nn.Module):
    def __init__(self, input_size, num_classes, number_hidden):
        super().__init__()

        ### Choose an activation function ###
        #self.act1 = nn.Tanh()
        #self.act1 = torch.sin
        #self.act1 = F.relu
        #self.act1 = nn.Sigmoid()
        self.act1 = Snake_split2
        #self.act1 = Snake_splitter
        
        self.linear1 = nn.Linear(input_size, number_hidden)
        self.linear2 = nn.Linear(number_hidden, number_hidden)
        self.linear3 = nn.Linear(number_hidden, number_hidden)
        self.linear5 = nn.Linear(number_hidden, number_hidden)
        self.linear4 = nn.Linear(number_hidden, num_classes, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        x = self.linear5(x)
        x = self.act1(x)
        x = self.linear4(x)
        return x

    


# Training and Predictions
train_loss_iterations = []
test_loss_iterations = []
preds_iter = []
# Run the experiment N times
for k in range(0, 5):
    random_seed = torch.get_rng_state()
    model = Net(input_size, num_classes, number_hidden)
    #Normalization and train + test split
    X_full = torch.Tensor(range(1, len(acc_list)+1)
                          ).float().unsqueeze(1)  # [1...7042]
    print('X_full len: ', len(X_full),
          '. X_full head (pre norm): ', X_full[:10])

    C = torch.max(X_full)
    X_full = X_full / C
    # Train Set input
    X = X_full[:len(X_full) - validation_size]
    print('X train len: ', len(X), '. X train head (post norm): ', X[:10])

    # Test Set input
    X_t = X_full[len(X_full) - validation_size:]
    print('X_t test len: ', len(X_t),
          '. X_t test head (post norm): ', X_t[:10])

    Y_full = torch.Tensor(acc_list[:]).unsqueeze(1)  # [13.45 ..... 233.37 ]
    print('Y_full len: ', len(Y_full),
          '. Y_full head (pre norm): ', Y_full[:10])

    Z = torch.max(Y_full)
    Y_full = Y_full / Z
    # Train Set output
    Y = Y_full[:len(X_full) - validation_size]
    print('Y train len: ', len(Y), '. Y train head (post norm): ', Y[:10])

    # Test Set output
    Y_t = Y_full[len(X_full) - validation_size:]
    print('Y_t test len: ', len(Y_t),
          '. Y_t test head (post norm): ', Y_t[:10])

    #Training
    test_losses = []
    train_losses = []

    opt = torch.optim.Adam(model.parameters(), learning_rate)
    loss_fn = F.mse_loss
    time_1 = time.time()
    for i in range(num_epochs):
      time_1 = time.time()

      opt.zero_grad()
      y = model(X)
      loss = loss_fn(y, Y)
      loss.backward()
      opt.step()
      time_2 = time.time()

      # Every 10 epochs (and last):
      # 1) print loss from training
      # 2) Run validation
      # 3) print loss from validation
      if i > 30 and (i % 10 == 0 or i == num_epochs - 1):
        train_losses.append(loss.item())

        # Run validation
        opt.zero_grad()
        y_validation = model(X_t)
        loss_v = loss_fn(y_validation, Y_t)
        test_losses.append(loss_v.item())
        print(
            f'Epoch: {i} --> Trainign Loss ---- {loss.item()}---- Test Loss ---- {loss_v.item()}----')

      # Every 100 iterations (and last)
      # 1) plot results from predictions: Train and Test
      # 2) plot ground truth: Train and Test
      if i % 100 == 0 or i == num_epochs - 1:
        #Produce predictions
        time_3 = time.time()
        y = model(X)
        y_validation = model(X_t)
        time_4 = time.time()
        y_concat = torch.cat((y, y_validation))

        plt.xlabel('Days', weight='bold', size=15)
        plt.ylabel('Temperature Measurements (extended)',
                   weight='bold', size=15)
        cut = len(y)
        plt.axvline(cut)
        plt.axis(ymin=-0.5, ymax=1)
        plt.title(
            f"Predictions using {model_name} and {study_name} .", weight='bold', size=15)
        plt.plot(y_concat.detach().numpy(), label='Predictions',
                 color='green')  # Validation loss == Blue
        # Training loss == orange
        plt.plot(Y_full, label='Ground Truth', color='orange', alpha=0.65)
        plt.legend(loc=2, prop={'size': 15})
        plt.savefig(
            f'./Tests_MPlus/Test_{k}/{i}_epochs_Predictions_{model_name}_{study_name}.png')
        plt.show(block=False)
        plt.close()

        # Zoomed in predictions on test
        plt.rcParams['figure.figsize'] = [13, 3]

        plt.xlabel('Days', weight='bold', size=15)
        plt.ylabel('Temperature Measurements (extended)',
                   weight='bold', size=15)
        plt.title(
            f"Test Set Predictions using {model_name} and {study_name}.", weight='bold', size=15)
        plt.plot(range(len(X)-validation_size, len(X)), y_validation.detach().numpy(), label='Predictions',
                 color='green')  # Validation loss == Blue
        # Training loss == orange
        plt.plot(range(len(X)-validation_size, len(X)), Y_t,
                 label='Ground Truth', color='orange', alpha=0.65)
        plt.legend(loc=2, prop={'size': 15})
        plt.savefig(
            f'./Tests_MPlus/Test_{k}/Test_only_{i}_epochs_Predictions_{model_name}_{study_name}.png')
        plt.show(block=False)
        plt.close()
        plt.rcParams['figure.figsize'] = [13, 6]

        if i == num_epochs - 1:
          #preds_iter.append(y_validation.detach().numpy())
          preds_iter.append(y_concat.detach().numpy())
    cut = len(y)
    train_loss_iterations.append(train_losses)
    test_loss_iterations.append(test_losses)

    # Plots for convergence of test and train error

    #train_losses = np.log(train_loss_iterations[0])
    #test_losses = np.log(test_loss_iterations[0])
    plt.rcParams['figure.figsize'] = [8, 5]
    for id, iteration in enumerate(train_loss_iterations):
      plt.xlabel('Iteration')
      plt.ylabel('Loss')
      plt.axis(ymin=-0, ymax=0.15)
      plt.title(
          f"Convergence over {num_epochs} epochs using {model_name} and {study_name}")
      plt.plot(test_losses, label='Test',
               color='green')  # Validation loss == Blue
      # Training loss == orange
      plt.plot(train_losses, label='Train', color='orange')
      plt.savefig(
          f'./Tests_MPlus/Test_{k}/convergence_after_{num_epochs}_epochs_{model_name}_{study_name}_{learning_rate}_{a_param}.png')
      plt.show(block=False)
      plt.close()
      time_training = (time_2 - time_1)*1000 * num_epochs
      time_inference = (time_4 - time_3)*1000
      textfile = open(
          f"./Tests_MPlus/Test_{k}/Seed.txt", "w")
      textfile.write("Random Seed: " + "\n" + str(random_seed))
      textfile.write("\n")
      textfile.close()

      textfile = open(
          f"./Tests_MPlus/Test_{k}/Preds_{model_name}_TM_{study_name}_{learning_rate}.txt", "w")
      textfile.write("predictions_train: " + "\n")
      for elem in y.detach().numpy():
        textfile.write(str(elem) + " ")
      textfile.write("\n")
      textfile.write("predictions_test: " + "\n")
      for elem in y_validation.detach().numpy():
        textfile.write(str(elem) + " ")
      textfile.write("\n")
      textfile.write("time_training: " + "\n" + str(time_training))
      textfile.write("\n")
      textfile.write("time_inference: " + "\n" + str(time_inference))
      textfile.write("\n")
      string1 = train_losses[-1]
      string2 = test_losses[-1]
      textfile.write("Train Loss: " + "\n" + str(string1))
      textfile.write("\n")
      textfile.write("Test_Loss: " + "\n" + str(string2))
      textfile.write("\n")
      textfile.close()

    textfile = open(
        f"./Tests_MPlus/Test_{k}/Seed.txt", "w")
    textfile.write("\n")
    textfile.write("Random Seed: " + "\n")
    for elem in random_seed:
      textfile.write(str(elem) + ' ')
    textfile.write("\n")
    textfile.close()
### Gather meta data ###
time_training = (time_2 - time_1)*1000 * num_epochs
time_inference = (time_4 - time_3)*1000
predictions_train = y.detach().numpy()
predictions_test = y_validation.detach().numpy()
test_loss = test_loss_iterations
train_loss = train_loss_iterations

textfile = open(
    f"./Tests_MPlus/Metadata_{model_name}_{study_name}_{learning_rate}.txt", "w")
textfile.write("time_training: " + "\n" + str(time_training))
textfile.write("\n")
textfile.write("time_inference: " + "\n" + str(time_inference))
textfile.write("\n")
textfile.write("predictions_train: " + "\n")
for elem in predictions_train:
  textfile.write(str(elem) + " ")
textfile.write("\n")
textfile.write("predictions_test: " + "\n")
for elem in predictions_test:
  textfile.write(str(elem) + " ")
textfile.write("\n")
string1 = ''
for iter in train_loss_iterations:
  string1 = string1 + ' ' + str(iter[-1])
string2 = ''
for iter in test_loss_iterations:
  string2 = string2 + ' ' + str(iter[-1])
textfile.write("train_losses: " + "\n" + string1)
textfile.write("\n")
textfile.write("test_losses : " + "\n" + string2)
textfile.write("\n")
textfile.write("Random Seed: " + "\n" + str(random_seed))
textfile.write("\n")
for elem in random_seed:
  textfile.write(str(elem.item()) + " ")

textfile.write("\n")
textfile.close()


bool1 = 0.35
bool2 = 0.35
bool3 = 0.35
bool4 = 0.35
bool5 = 0.35
bools = [bool1, bool2, bool3, bool4, bool5]
testers = []
best = 1000
best_id = -1
for id, elem in enumerate(test_loss_iterations):
  testers.append(elem[-1])
  if elem[-1] < best:
    best_id = id
    best = elem[-1]
bools[best_id] = 1
print('Testers: ', str(testers), '  - Best one is: ', best_id, ' ', best)
#Special Data
plt.rcParams['figure.figsize'] = [13, 6]
plt.xlabel('Days', weight='bold', size=15)
plt.ylabel('Temperature Measurements (extended)', weight='bold', size=15)
cut = len(X)
plt.axis(ymin=-0.5, ymax=1)
plt.axvline(cut)
plt.title(
    f"Predictions using {model_name} using {study_name} .", weight='bold', size=15)
plt.plot(y.detach().numpy(), color='green')  # Validation loss == Blue
plt.plot(Y_full, color='orange', alpha=0.65)
# Training loss == orange
plt.plot(
    #range(len(X) , len(X) + len(preds_iter[0]) ),
    preds_iter[0], label='Test 1', color='green', alpha=bools[0])
plt.plot(
    #range( len(X) , len(X) + len(preds_iter[0]) ),
    preds_iter[1], label='Test 2', color='green', alpha=bools[1])
plt.plot(
    #range( len(X) , len(X) + len(preds_iter[0]) ),
    preds_iter[2], label='Test 3', color='green', alpha=bools[2])
plt.plot(
    #range( len(X) , len(X) + len(preds_iter[0]) ),
    preds_iter[3], label='Test 4', color='green', alpha=bools[3])
plt.plot(
    #range( len(X) , len(X) + len(preds_iter[0]) ),
    preds_iter[4], label='Test 5', color='green', alpha=bools[4])
plt.legend(loc=2, prop={'size': 8})
plt.savefig(
    f'./Tests_MPlus/Predictions_{model_name}_{study_name}.png')
plt.show(block=False)
plt.close()


avg_train_loss = []
for elem in train_loss_iterations:
  avg_train_loss.append(elem[-1])
avg_val1 = sum(avg_train_loss) / len(avg_train_loss)
st_dev1 = np.std(avg_train_loss)

avg_test_loss = []
for elem in test_loss_iterations:
  avg_test_loss.append(elem[-1])
avg_val2 = sum(avg_test_loss) / len(avg_test_loss)
st_dev2 = np.std(avg_test_loss)

textfile = open(
    f"./Tests_MPlus/Metadata_{model_name}_{study_name}_{learning_rate}.txt", "w")
textfile.write("Avg Train: " + "\n" + str(avg_val1))
textfile.write("\n")
textfile.write("STD Train: " + "\n" + str(st_dev1))
textfile.write("\n")
textfile.write("AVG Test: " + "\n" + str(avg_val2))
textfile.write("\n")
textfile.write("STD Test: " + "\n" + str(st_dev2))
textfile.write("\n")
textfile.close()
