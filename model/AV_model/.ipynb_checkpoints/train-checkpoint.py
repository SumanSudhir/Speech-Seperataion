from __future__ import print_function
import sys
sys.path.append('../lib')
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import pickle
from model import AV_model
from dataloader import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--LR', type=float)
parser.add_argument('--number_of_people', type=int)
parser.add_argument('--restart_training', type=str)
parser.add_argument('--path', type=str)
args = parser.parse_args()

# parameters and hyper params

# can be set by terminal
batch_size = args.batch_size
number_of_people = args.number_of_people
LR = args.LR

# set manually 
val_iterations = (30000//(batch_size))//10     #using only 3.33% of validation data
train_iterations =  (70000*5//(batch_size))//1  #using only 10% of training data  (5 epochs)
# val_iterations = 52

model_checkpoint_dir = 'checkpoint_dir'
plot_main_dir = 'Plots'
plots_dir = 'Plots/fig'
plots_pickle_dir = 'Plots/pickle' 

# print_every = 100
# save_every = 100
# valid_every = 1200

print_every = 116
save_every = 116
valid_every = 232

# print_every = 5

if args.restart_training == 'true':
    if os.path.isdir(model_checkpoint_dir):
        shutil.rmtree(model_checkpoint_dir, ignore_errors=True)
    if os.path.isdir(plot_main_dir):
        shutil.rmtree(plot_main_dir, ignore_errors=True)
    
    os.mkdir(model_checkpoint_dir)
    os.mkdir(plot_main_dir)
    os.mkdir(plots_dir)
    os.mkdir(plots_pickle_dir)

elif args.restart_training == 'false':
    
    if not os.path.isdir(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    if not os.path.isdir(plot_main_dir):
        os.mkdir(plot_main_dir)
    
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    
    if not os.path.isdir(plots_pickle_dir):
        os.mkdir(plots_pickle_dir)
    
    
# gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status =', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# using seed so to be deterministic
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(1)

        
def main():
    # initialise the models

    net = AV_model(number_of_people).double().to(device)
#     net.apply(init_weights)
    
    train_loader = DataLoader(train=True, number_of_people=number_of_people, shuffle=False, batchsize=batch_size,
                              dataset_loc='../../data/dataset', device=device)
    valid_loader = DataLoader(train=False, number_of_people=number_of_people, shuffle=False, batchsize=batch_size,
                              dataset_loc='../../data/dataset', device=device)

    data_size = train_loader.data_file.shape[0]
    data_size_valid = valid_loader.data_file.shape[0]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    loss_list_train = [None]
    index_data_list = [None]
    loss_list_validation = [None]
    loss_list_validation_index = [None]
    
    train_iterations_old = 0
    
    if args.restart_training == 'false':
        checkpoint = torch.load(args.path, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['model_state_dict'])
        train_iterations_old = checkpoint['iterations']     
        
        if checkpoint['train_loss'] == None:
            loss_list_train = [None]
            index_data_list = [None]
        else:
            loss_list_train.append(checkpoint['train_loss'])  
            index_data_list.append(checkpoint['iterations']*batch_size)     
        
        if checkpoint['valid_loss'] == None:
            loss_list_validation = [None]
            loss_list_validation_index = [None]
        else:
            loss_list_validation.append(checkpoint['valid_loss'])     
            loss_list_validation_index.append(checkpoint['iterations'])
    net.train()       
    for index in range(train_iterations_old, train_iterations):
        
        if (index+1)%10 == 0:
#             print(index%(train_loader.data_file.shape[0]//batch_size))
            print('Index:', index+1, ' Time:', datetime.datetime.now())
            
        running_loss = 0.0
        train_count = 0
        
        audio_input, video_input, target_label = train_loader[index%(train_loader.data_file.shape[0]//batch_size)]
        optimizer.zero_grad()
        output = net(audio_input, video_input)
        loss = criterion(target_label, output)
        loss.backward()
        optimizer.step()
            
        running_loss += loss.item()
        train_count += 1
        
        if (index+1) % print_every == 0:  # print every print_every mini_batch of data
            print('Index:%d Loss:%.6f' %
                    (index + 1, running_loss / train_count),' Time:',datetime.datetime.now())
                
            index_data_list.append((index+1)*batch_size)
            loss_list_train.append(running_loss / train_count)
            running_loss = 0.0
            train_count = 0
                
            plt.plot(index_data_list[1:], loss_list_train[1:], label = "Training", color='green', marker='o', markerfacecolor='blue', markersize=5)
            plt.xlabel('Data encountered') 
            plt.ylabel('Loss') 
            plt.savefig(plots_dir + '/train_plot.png')
            plt.clf()
            
            training_pickle = open(plots_pickle_dir + "/loss_list_train.npy",'wb')
            pickle.dump(loss_list_train,training_pickle)
            training_pickle.close()
        
        if (index+1) % save_every == 0:
            print('Saving model at %d index' % (index + 1),' Time:',datetime.datetime.now())  # save every save_every mini_batch of data
            torch.save({
            'iterations': index+1,
            'batchsize': batch_size,
            'train_loss': loss_list_train[-1],
            'valid_loss': loss_list_validation[-1],
            'model_state_dict': net.state_dict(),
            }, model_checkpoint_dir + '/model_%d.pth' % (index + 1))
            
#         if (index+1) % valid_every == 0:
#             net.eval()  
#             total_loss_valid = 0
#             for valid_index in range(val_iterations):
#                 with torch.no_grad():
#                     audio_input, video_input, target_label = valid_loader[valid_index%(valid_loader.data_file.shape[0]//batch_size)]
#                     output = net(audio_input, video_input)
#                     loss = criterion(target_label, output)
#                     total_loss_valid += loss.item()

#             loss_list_validation.append(total_loss_valid / val_iterations)
#             loss_list_validation_index.append(index+1)
            
#             net.train()
#             print('Index %d Valid Loss: %.3f' % (index + 1, loss_list_validation[-1]),' Time:',datetime.datetime.now() )

#             plt.plot(loss_list_validation_index[1:], loss_list_validation[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
#             plt.xlabel('Iteration') 
#             plt.ylabel('Validation Loss') 
#             plt.savefig(plots_dir + '/valid_plot.png')
#             plt.clf()
            
#             validation_pickle = open(plots_pickle_dir + "/loss_list_validation.npy",'wb')
#             pickle.dump(loss_list_validation,validation_pickle)
#             validation_pickle.close()

#             if len(loss_list_validation) >= 3:
#                 if loss_list_validation[-1] < loss_list_validation[-2]:
#                     torch.save({
#                     'iterations': index + 1,
#                     'batchsize': batch_size,
#                     'train_loss': loss_list_train[-1],
#                     'valid_loss': loss_list_validation[-1],
#                     'model_state_dict': net.state_dict(),
#                     }, model_checkpoint_dir + '/best_model.pth')
#             else:
#                 torch.save({
#                 'iterations': index + 1,
#                 'batchsize': batch_size,
#                 'train_loss': loss_list_train[-1],
#                 'valid_loss': loss_list_validation[-1],
#                 'model_state_dict': net.state_dict(),
#                  }, model_checkpoint_dir + '/best_model.pth')

                
                
if __name__ == '__main__':
    main()
