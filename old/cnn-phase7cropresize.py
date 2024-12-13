#!/usr/bin/env python3
# CNN on phase images
# with cropping and resizing
# 16/11/2024

# Inputs are npy-files: measured_phase.py
# with dim: 1200x 1920
# and min: -PI and max: +PI
# xcopy .\*_phase.npy C:\Users\LFC_01\destination\ /S /C /Y

#%% Imports
import tensorflow as tf
print("\nTensorflow and Keras versions:")
print(tf.__version__)
print(tf.keras.__version__)
from keras import datasets, layers, models
from keras.losses import SparseCategoricalCrossentropy
# import sklearn
from sklearn.metrics import classification_report
# other imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import time   # compute elapsed time


#%% Parameters
# ################################################################ # 
# Set main parameters                                              #
# ################################################################ #

# ### Parameters ###
# Number of epochs
n_epochs = 10
# Set training set percentage
train_set_percentage = 0.5

# Crop images parameters
n_crops_per_image = 10  # number of crops per image
x_crop_size = 800  # number of pixels for x axis (width max 1920)
y_crop_size = 600  # number of pixels for y axis (high max 1200)
tf.random.set_seed(1234)  # initialize random seed for tensorflow
# Resize images parameters
x_resize = 256  # width
y_resize = 256  # high
# ###########################

# Number of runs of measures
data_acquisition_runs = 60

# Data path
# data_path = r'C:\Users\LFC_01\Desktop\Szilard\Data'
data_path = r'/Volumes/EXTERNAL_US/2024_06_12/'
# data_path = r'data3'

# Date of measurements
# date='2024_04_04'
date=''
    
# Define the type of measure, so we can read the proper folder
data_type_measure = "SLM_Interferometer_alignment"

# Classes
classes=["input_mask_blank_screen", 
         "input_mask_horizontal_division_A", "input_mask_horizontal_division_B", 
         "input_mask_vertical_division_A", "input_mask_vertical_division_B",
         "input_mask_checkboard_1A", "input_mask_checkboard_1B", "input_mask_checkboard_1R", 
         "input_mask_checkboard_2A", "input_mask_checkboard_2B", 
         "input_mask_checkboard_3A", "input_mask_checkboard_3B",
         "input_mask_checkboard_4A", "input_mask_checkboard_4B",]
classes_short=np.arange(np.size(classes))

# Measured Phase file names
measured_phase_input_file = "measured_phase.npy"

# Number of frames
frame_number = data_acquisition_runs * np.size(classes)

# Print values
print("\n=================================================================")
print("Input values: ")
print(" - n_crops_per_image: ", n_crops_per_image)
print(" - x_crop_size: ", x_crop_size)
print(" - y_crop_size: ", y_crop_size)
print(" - x_resize", x_resize)
print(" - y_resize", y_resize)
print(" - Number of Runs of measures: ", data_acquisition_runs)
print(" - Classes in words:", classes)
print(" - Classes in numbers: ", classes_short)
print(" - Total number of classes: ", np.size(classes))
print(" - Measured data | frame_number (number of RUNs * number of classes): ", frame_number)
print("=================================================================")


#%% Determine high and widht of the matrix reading the data
print("\n### Start of reading data ###")
sample_measured_phase_full_path = os.path.join(data_path, date, data_type_measure, "M00001", classes[0], "files", measured_phase_input_file)
sample_measured_phase = np.load(sample_measured_phase_full_path)
high, width = sample_measured_phase.shape
print("\nLoaded sample phase with dimensions in px (high, width): ", high, width)


#%% Load data and plot

# initialization of the frame matrix
measured_phase = np.zeros((frame_number, high, width))
reference_class_short = np.zeros((frame_number, 1), dtype=int)

j=0 #  index for frame_number (varies from 0 to frame_number-1)
for i in range(data_acquisition_runs):
    RUN = 'M0000' + str(i+1)
    if i+1 > 9:
        RUN = 'M000' + str(i+1)
        if i+1 > 99:
            RUN = 'M00' + str(i+1)
    k=0  # index for classes (varies from 0 to )
    for class_name in classes :
        input_files_path = os.path.join(data_path, date,data_type_measure, RUN, class_name, "files", measured_phase_input_file)
        measured_phase[j,:,:] = np.load(input_files_path)
        reference_class_short[j] = k
        # ### Test print values of the measured phase
        # print("\nShape of measured phase: ", measured_phase.shape)
        # print(reference_class_short.shape)
        # print('min of measured_phase: ', measured_phase.min())
        # print('max of measured_phase: ', measured_phase.max())

        k+=1
        j+=1

# # Plot and Verify the data
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(measured_phase[i+222])
#     plt.xlabel(int(reference_class_short[i+222]))
# plt.show()
# sys.exit()
# # End of plot

# Print arrays dimensions
print("- measured_phase tensor shape: ", tf.shape(measured_phase).numpy())
print("- reference_class_short tensor shape: ", reference_class_short.shape)


# #%% Crop images
print("\n### Start of cropping ###")

# Start the timer
start_time = time.time()

# Initialize arrays to store cropped images and labels
cropped_measured_phase = []
cropped_reference_class_short = []

# Perform random cropping
for i in range(frame_number):
    for _ in range(n_crops_per_image):
        cropped_image = tf.image.random_crop(measured_phase[i], size=[y_crop_size, x_crop_size])
        cropped_measured_phase.append(cropped_image)
        cropped_reference_class_short.append(reference_class_short[i])

# Convert lists to numpy arrays
cropped_measured_phase = np.array(cropped_measured_phase)
cropped_reference_class_short = np.array(cropped_reference_class_short)

# Update frame_number to reflect the new number of images
frame_number = len(cropped_measured_phase)
print("- cropped_measured_phase tensor shape: ", tf.shape(cropped_measured_phase).numpy())
print("- cropped_reference_class_short tensor shape: ", cropped_reference_class_short.shape)
print("- New frame_number: ", frame_number)
 
# # Plot and Verify the data
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(measured_phase[i+620])
#     plt.xlabel(int(reference_class_short[i+620]))
# plt.show()
# sys.exit()
# # end of plot

# End the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken for the cropping: {elapsed_time:.2f} seconds")

#%% Resize images
print("\n### Start of resizing ###")
print("measured_phase.shape: ", measured_phase.shape)
print("measured_phase[0].shape: ", measured_phase[0].shape)
print("cropped_measured_phase.shape: ", cropped_measured_phase.shape)
print("cropped_measured_phase[0].shape: ", cropped_measured_phase[0].shape)
print("cropped_reference_class_short.shape: ", cropped_reference_class_short.shape)
print("cropped_reference_class_short[0].shape: ", cropped_reference_class_short[0].shape)
print("frame_number: ", frame_number)

resize_measured_phase = np.zeros((frame_number, y_resize, x_resize))

for ii in range(frame_number):
    resize_measured_phase[ii] = cv2.resize(cropped_measured_phase[ii], (y_resize, x_resize), interpolation = cv2.INTER_NEAREST)

# reassign values just to make things works simplier
cropped_measured_phase = resize_measured_phase


#%% Split data between Train and Test, then Normalization
# TODO: Clever split with tensorflow
print("\n### Splitting data ###")
print("train_set split number: ", np.int32(frame_number * train_set_percentage))
print("test_set split number: ", frame_number-np.int32(frame_number*train_set_percentage))

# Split data between Train and Test
train_measured_phase, test_measured_phase = tf.split(cropped_measured_phase, [np.int32(frame_number * train_set_percentage), frame_number - np.int32(frame_number * train_set_percentage)])
train_reference_class_short, test_reference_class_short = tf.split(cropped_reference_class_short, [np.int32(frame_number * train_set_percentage), frame_number - np.int32(frame_number * train_set_percentage)])

print("- train_measured_phase tensor shape: ", tf.shape(train_measured_phase).numpy())
print("- test_measured_phase tensor shape: ", tf.shape(test_measured_phase).numpy())
print("- train_reference_class_short tensor shape: ", tf.shape(train_reference_class_short).numpy())
print("- test_reference_class_short tensor shape: ", tf.shape(test_reference_class_short).numpy())

# Normalize "pixel" values to be between 0 and 1
train_measured_phase = (train_measured_phase - np.min(train_measured_phase)) / (np.max(train_measured_phase)-np.min(train_measured_phase))
test_measured_phase = (test_measured_phase - np.min(test_measured_phase)) / (np.max(test_measured_phase)-np.min(test_measured_phase))

# # Plot and Verify the data
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_measured_phase[i])
#     plt.xlabel(int(train_reference_class_short[i]))
# plt.show()
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_measured_phase[i+620])
#     plt.xlabel(int(train_reference_class_short[i+620]))
# plt.show()
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_measured_phase[i])
#     plt.xlabel(int(test_reference_class_short[i]))
# plt.show()
# sys.exit()
# # End of plot

#%% Create the convolutional base
print("\n### Create neurons ###")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(y_resize, x_resize, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Display the model
model.summary()

#%% Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(np.size(classes)))
# Display the model
model.summary()

#%% Compile and train the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_measured_phase, train_reference_class_short, epochs=n_epochs, 
                    validation_data=(test_measured_phase, test_reference_class_short))


#%% Evaluate the model: plot training and validation accuracy
plt.figure('Training and validation accuracy')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

# Evaluate the model: print the test accuracy
test_loss, test_acc = model.evaluate(test_measured_phase, test_reference_class_short, verbose=2)
print("test accuracy: ", test_acc)

# Predict the classes
predictions = model.predict(test_measured_phase)
predicted_classes = np.argmax(predictions, axis=1)

# Print classification report
report = classification_report(
    test_reference_class_short, predicted_classes, target_names=classes, digits=6,
    output_dict=False)
print(report)

# Plot the training and validation loss
plt.show()
sys.exit(0)
