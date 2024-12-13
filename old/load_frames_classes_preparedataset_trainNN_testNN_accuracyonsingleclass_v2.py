# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:02:44 2023

@author: LFC_01
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# path_file= r'C:\Users\LFC_01\Desktop\Szilard\Data\2023_09_28\SLM_Interferometer_alignment\M00001\input_mask_blank_screen\files\measured_phase.npy'
# measured_phase=np.load (path_file)

#%% FUNCTIONS
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def normtensorframe(bachframes):
    #phase values inbetween [-pi,pi] are normalized to [0,1]
    sampleimagenorm=torch.zeros(bachframes.shape)
    for i in range(bachframes.shape[0]):
        sampleimagenorm[i,:,:]=(bachframes[i,:,:]+np.pi)/(2*np.pi)    
    return sampleimagenorm

   
def imshowongrid(imgtensor):
    N=imgtensor.shape[0]
    fig, axes=plt.subplots(1,N)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between subplots
    for i, ax in enumerate(axes.flat):
        img = imgtensor.numpy()[i,:,:]*2*np.pi - np.pi     # unnormalize
        ax.imshow(img)
        ax.axis('off')
    
#    cbar = fig.colorbar(im, cax=cbar_axes)
    
#%% CLASSES
class CustomDataset(Dataset):
    def __init__(self, data, labels):#, transform=None):
        self.data = data
        self.labels = labels
        #self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

        # if self.transform:
        #     sample = self.transform(sample)

        return sample

#%%
# General data path
data_path = r'C:\Users\LFC_01\Desktop\Szilard\Data'
data_path = r'data'

#Date of measurements
date='2024_04_04'
date=''
    
# Define the type of measure, so we can make a folder
data_type_measure = "SLM_Interferometer_alignment"

#Number of RUN
nRUN=10

#Classes
classes=["input_mask_blank_screen", 'input_mask_vertical_division', 'input_mask_horizontal_division', 'input_mask_checkboard_1', 'input_mask_checkboard_2', 'input_mask_checkboard_3', 'input_mask_checkboard_4']
classes_short=np.arange(np.size(classes))

#Number of frames
Nframes=nRUN*np.size(classes)

#Data to load
data_name=os.path.join('files','measured_phase.npy')
test_dir_path=os.path.join(data_path, date,data_type_measure, 'M00001', classes[0], data_name)
testmatrix=np.load(test_dir_path)
H, W=testmatrix.shape


#Inizialization of the frame matrix
measured_phase=np.zeros((H,W,Nframes))
reference_class_short=np.zeros((Nframes,1), dtype=int)
#%% DATA LOAD and plot
j=0
for i in range(nRUN):
    RUN='M0000'+ str(i+1)
    if i+1>9:
        RUN='M000'+ str(i+1)
        if i+1>99:
            RUN='M00'+ str(i+1)
    k=0
    for class_name in classes :
        dir_path=os.path.join(data_path, date,data_type_measure, RUN, class_name, data_name)
        measured_phase[:,:,j]=np.load(dir_path)
        reference_class_short[j]=k
        # #figure
        # img=plt.imshow(measured_phase[:,:,j])#cmap='Blues'
        # plt.axis('off')
        # plt.colorbar(img)
        # plt.show()
        # plt.pause(0.1)
        # print(dir_path)
        # print(j)
        #
        k+=1
        j+=1
        

#%%DATA AUGUMENTATION THROUGH CROPPING AND BINNING

# I define crop size as integer multiple Nc of the size of CIFAR-10 images, then downsize the image resolution by a factor Nc by pixel binning of. This way it is possible to feed the data to the conv net developed for classifying CIFAR-10
 
Sc=32 #(px) use Sc=32 to have the size of  CIFAR-10 images
Nc=20

#define crop size
wc=hc=Sc*Nc #(px)

#available corner points intervals
xc_0, yc_0=[0,0]
xc_end, yc_end=[W-wc, H-hc]

#number of randomly selected cornerpoints
Nc=4*60 #select an integer multiple of the batch size

#corner points
xc=random.choices(range(xc_0, xc_end), k=Nc)
yc=random.choices(range(yc_0, yc_end), k=Nc)

#intialize crop matrix
Nframescrop=Nframes*Nc

#crop shape after binning
# newshape=np.array([np.round(wc/10), np.round(hc/10)])
# newshape=newshape.astype(int)
newshape=np.array([Sc, Sc])

measured_phase_crop=np.zeros((Nframescrop,newshape[0],newshape[1]))
reference_class_short_crop=np.zeros((Nframescrop,1), dtype=int)

#create 3D matrix of cropped frames
k=0
for i in range (Nframes):
    #figure
    # img=plt.imshow(measured_phase[:,:,i])#cmap='Blues'
    # plt.axis('off')
    # plt.colorbar(img)
    # plt.show()
    # plt.pause(0.1)
    for j in range(Nc):
        crop=measured_phase[yc[j]:yc[j]+hc, xc[j]:xc[j]+wc ,i]
        crop_binned=rebin(crop,newshape)
        measured_phase_crop[k,:,:]=crop_binned
        reference_class_short_crop[k]=reference_class_short[i]
        #figure
        # img=plt.imshow(measured_phase_crop[k,:,:])#cmap='Blues'
        # plt.axis('off')
        # plt.colorbar(img)
        # plt.show()
        # plt.pause(0.1)
        if k % 1000==999:
            print(k+1)
        #
        k+=1

print(k)
# #%%TESTS
# phase=measured_phase_crop[1004,:,:]
# img=plt.imshow(phase)#cmap='Blues'
# plt.axis('off')
# plt.colorbar(img)
# plt.show()

#%% CREATE DATASET FOR PYTORCH

# Create your custom dataset
phase_wrapped_dataset = CustomDataset(measured_phase_crop, reference_class_short_crop)
# Split the dataset into training (90%) and testing (10%) dataset
dataset_size=measured_phase_crop.shape[0]
trainset_size=np.int32(0.9*dataset_size)
testset_size=np.int32(0.1*dataset_size)
phase_wrapped_train_dataset, phase_wrapped_test_dataset = torch.utils.data.random_split(phase_wrapped_dataset, (trainset_size,testset_size))

# Create a DataLoader to handle batching, shuffling, and more
batch_size = 4
phase_wrapped_train_dataloader = DataLoader(phase_wrapped_train_dataset, batch_size=batch_size, shuffle=True)
phase_wrapped_test_dataloader = DataLoader(phase_wrapped_test_dataset, batch_size=batch_size, shuffle=True)

# get some random training images
dataiter = iter(phase_wrapped_train_dataloader)
sample= next(dataiter)
batch_images=normtensorframe(sample['data'])#phase values normilized to [0,1]
batch_labels=sample['label']

#show bathc images and print labels
imshowongrid(batch_images)
print('Labels:\n','\n'.join(f'{ classes[batch_labels.numpy()[:,0][i]]}' for i in range(batch_size) ))

#%% NEURAL NETWORK
n_out_classes=classes_short.shape[0] #number of output classes
n_in_imgchannel=1 #set n_imgchannel=1 for grayscale images, set n_imgchannel=3 for color images+

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in_imgchannel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_out_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#%% LOSS FUNCTION AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #Application of stochastic gradient descend (SGD) through the package torch.optim

    #%% TRAINING 
Nepoch=5   
for epoch in range(Nepoch):  # loop over the dataset multiple times
#epoch=0
    running_loss = 0.0
    for i, sample in enumerate(phase_wrapped_train_dataloader, 0):
        
        # INPUT
        input=normtensorframe(sample['data'])#phase values normalized to [0,1]
        input=input[:,None,:,:]
        labels=sample['label']
        labels=torch.squeeze(torch.Tensor.long(labels))
        #show batch images and print labels
        #imshowongrid(torch.squeeze(input))
        #print('labels:', ' '.join(f'{ labels.numpy()[i]}' for i in range(batch_size) ))
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # OUTPUT
        output = net(input)
        
        #LOSS FUNCTION
        loss = criterion(output, labels)
        #print(loss)
        
        # BACKPROPAGATION
        #Application of stochastic gradient descend (SGD) through the package torch.optim
        loss.backward()
        
        # UPDATE OF THE WEIGHTS
        optimizer.step()    # Does the update
        
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every  mini-batches
            running_loss_avg=running_loss/1000
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_avg:.3f}')
            running_loss = 0.0
        # running_loss = loss.item()
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')      
print('Finished Training')
#%% TEST SINGLE BATCH OF IMAGES
     
# get some random test images
dataiter = iter(phase_wrapped_test_dataloader)
sample= next(dataiter)
batch_images=normtensorframe(sample['data'])#phase values normilized to [0,1]
batch_labels=sample['label']
#show batc images and print labels
imshowongrid(batch_images)
print('\nGroundTruth:\n','\n'.join(f'{ classes[batch_labels.numpy()[:,0][i]]}' for i in range(batch_size) )) 

#test the conv net
input=normtensorframe(sample['data'])#phase values normalized to [0,1]
input=input[:,None,:,:]
# OUTPUT
output = net(input)
_, predicted = torch.max(output, 1) # get the index of the highest energy to find the predicted class
print('\nPredicted:\n', '\n'.join(f'{classes[predicted[j]]}' for j in range(4)))

#%% TESTING ON ENTIRE DATASET
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i, sample in enumerate(phase_wrapped_test_dataloader, 0):
       # INPUT
       input=normtensorframe(sample['data'])#phase values normalized to [0,1]
       input=input[:,None,:,:]
       labels=sample['label']
       labels=torch.squeeze(torch.Tensor.long(labels))
       # OUTPUT
       output = net(input)
        # the class with the highest energy is what we choose as prediction
       _, predicted = torch.max(output, 1)
       #print('\nPredicted:\n', '\n'.join(f'{classes[predicted[j]]}' for j in range(4)))
       total += labels.size(0)
       correct += (predicted == labels).sum().item()

#print(f'Accuracy of the network on the {testset_size} test images: {100 * correct // total}%')
print(f'\n\n Accuracy of the network: {100 * correct // total}%')
print('Parameters:\n',f'nRUN={nRUN}\n',f'crop size={wc} px\n',f'crop size after binning={Sc} px\n', f'N random crop={Nc}\n', f'N frames={Nframescrop}\n', f"trainset size={trainset_size}\n", f"testset size={testset_size}\n", f'batch size={batch_size}\n', f'N epoch={Nepoch}\n',f'running loss={running_loss_avg}\n' )

#%%TEST SINGLE CLASSES ON ENTIRE DATASET
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
 for i, sample in enumerate(phase_wrapped_test_dataloader, 0):
        # INPUT
        input=normtensorframe(sample['data'])#phase values normalized to [0,1]
        input=input[:,None,:,:]
        labels=sample['label']
        labels=torch.squeeze(torch.Tensor.long(labels))
        # OUTPUT
        output = net(input)
        
        _, predictions = torch.max(output, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

