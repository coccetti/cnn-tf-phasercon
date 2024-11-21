import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
import time
from keras import datasets, layers, models
from sklearn.metrics import classification_report
from tensorflow.keras.losses import SparseCategoricalCrossentropy

tf.random.set_seed(1234)  # initialize random seed for tensorflow

# Resize images parameters
x_resize = 128  # width
y_resize = 128  # high

# Data path
data_path = r'/Volumes/EXTERNAL_US/2024_06_12/'
# data_path = r'data3'

# Date of measurements
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
         "input_mask_checkboard_4A", "input_mask_checkboard_4B"]

# Loop over different nRUN values
for nRUN in range(30, 301, 10):
    print(f"\nRunning with nRUN = {nRUN}")

    # Number of frames
    frame_number = nRUN * np.size(classes)

    # Print values
    print("\n=================================================================")
    print("Input values: ")
    print(" - nRUN: ", nRUN)
    print(" - Classes in words:", classes)
    print(" - Classes in numbers: ", np.arange(np.size(classes)))
    print(" - Total number of classes: ", np.size(classes))
    print(" - Measured data | frame_number (number of RUNs * number of classes): ", frame_number)
    print("=================================================================")

    # Determine high and width of the matrix reading the data
    print("\n### Start of reading data ###")
    sample_measured_phase_full_path = os.path.join(data_path, date, data_type_measure, "M00001", classes[0], "files", "measured_phase.npy")
    sample_measured_phase = np.load(sample_measured_phase_full_path)
    high, width = sample_measured_phase.shape
    print("\nLoaded sample phase with dimensions in px (high, width): ", high, width)

    # Load data and plot
    measured_phase = np.zeros((frame_number, high, width))
    reference_class_short = np.zeros((frame_number, 1), dtype=int)

    j = 0  # index for frame_number (varies from 0 to frame_number-1)
    for i in range(nRUN):
        RUN = 'M0000' + str(i+1)
        if i+1 > 9:
            RUN = 'M000' + str(i+1)
            if i+1 > 99:
                RUN = 'M00' + str(i+1)
        k = 0  # index for classes (varies from 0 to )
        for class_name in classes:
            input_files_path = os.path.join(data_path, date, data_type_measure, RUN, class_name, "files", "measured_phase.npy")
            measured_phase[j,:,:] = np.load(input_files_path)
            reference_class_short[j] = k
            k += 1
            j += 1

    # Print arrays dimensions
    print("- measured_phase tensor shape: ", tf.shape(measured_phase).numpy())
    print("- reference_class_short tensor shape: ", reference_class_short.shape)

    # Resize images
    print("\n### Start of resizing ###")
    resize_measured_phase = np.zeros((frame_number, y_resize, x_resize))
    for ii in range(frame_number):
        resize_measured_phase[ii] = cv2.resize(measured_phase[ii], (y_resize, x_resize), interpolation=cv2.INTER_NEAREST)
    measured_phase = resize_measured_phase

    # Split data between Train and Test, then Normalization
    train_set_percentage = 0.5  # Set training set percentage
    print("\n### Splitting data ###")
    train_measured_phase, test_measured_phase = tf.split(measured_phase, [np.int32(frame_number * train_set_percentage), frame_number - np.int32(frame_number * train_set_percentage)])
    train_reference_class_short, test_reference_class_short = tf.split(reference_class_short, [np.int32(frame_number * train_set_percentage), frame_number - np.int32(frame_number * train_set_percentage)])
    print("- train_measured_phase tensor shape: ", tf.shape(train_measured_phase).numpy())
    print("- test_measured_phase tensor shape: ", tf.shape(test_measured_phase).numpy())
    print("- train_reference_class_short tensor shape: ", tf.shape(train_reference_class_short).numpy())
    print("- test_reference_class_short tensor shape: ", tf.shape(test_reference_class_short).numpy())

    # Normalize "pixel" values to be between 0 and 1
    train_measured_phase = (train_measured_phase - np.min(train_measured_phase)) / (np.max(train_measured_phase) - np.min(train_measured_phase))
    test_measured_phase = (test_measured_phase - np.min(test_measured_phase)) / (np.max(test_measured_phase) - np.min(test_measured_phase))

    # Create the convolutional base
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(y_resize, x_resize, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add Dense layers on top
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(np.size(classes), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_measured_phase, train_reference_class_short, epochs=10, 
                        validation_data=(test_measured_phase, test_reference_class_short))

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation Accuracy for nRUN = {nRUN}')
    
    # Evaluate the model: print the test accuracy
    test_loss, test_acc = model.evaluate(test_measured_phase, test_reference_class_short, verbose=2)
    print("test accuracy: ", test_acc)

    # Predict the classes
    predictions = model.predict(test_measured_phase)
    predicted_classes = np.argmax(predictions, axis=1)

    # Print classification report
    report = classification_report(test_reference_class_short, predicted_classes, target_names=classes, digits=6, output_dict=False)
    print(report)

    # Save classification report to file in Downloads folder
    downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    if not os.path.exists(downloads_folder):
        os.makedirs(downloads_folder)
    report_path = os.path.join(downloads_folder, f'classification_report_nRUN_{nRUN}.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Save plot to file in Downloads folder
    plt.savefig(os.path.join(downloads_folder, f'training_val_acc_{nRUN}nRUN_{x_resize}xres_{y_resize}yres.png'))
    plt.close()

    