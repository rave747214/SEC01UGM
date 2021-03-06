# Colab library to upload files to notebook
from google.colab import files
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Install Kaggle library
!pip install -q kaggle

# Upload kaggle API key file
data = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset from kaggle
!kaggle datasets download -d tawsifurrahman/covid19-radiography-database

# Extract zipfile
import zipfile
zip_ref = zipfile.ZipFile('covid19-radiography-database.zip', 'r')
zip_ref.extractall('files')
zip_ref.close()

# Modules for train-val split
import os
import numpy as np
import random
import argparse
from shutil import copyfile


# Train-val split function
def img_train_test_split(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    else:
        if not os.path.exists('data/train'):
            os.makedirs('data/train')
        if not os.path.exists('data/validation'):
            os.makedirs('data/validation')

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('data/train', subdir)
        validation_subdir = os.path.join('data/validation', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                fileparts = filename.split('.')

                if random.uniform(0, 1) <= train_size:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
                    train_counter += 1
                else:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
                    validation_counter += 1

        print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)

        
# Remove unused folder
!rm -rf '/content/files/COVID-19 Radiography Database/Viral Pneumonia' 

# Run the split function
img_train_test_split('/content/files/COVID-19 Radiography Database',0.7)

!pip install -q pyyaml h5py

## Baseline
# Modules for model creation
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall

prec = Precision(name='prec')
rec = Recall(name='rec')

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',prec,rec])

model.summary()

# Label generator
TRAINING_DIR = "/content/data/train/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(300, 300))

VALIDATION_DIR = "/content/data/validation/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=10,
                                                              class_mode='binary',
                                                              target_size=(300, 300))

# Fit the model
history = model.fit(train_generator,
                              epochs=100,
                              verbose=1,
                              validation_data=validation_generator)

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
def plot_ax(axes, metrics, title, epochs):
    val_met = 'val_' + metrics
    axes.plot(history.history[metrics], c='blue', linewidth=1.5)
    axes.plot(history.history[val_met], c='orange', linewidth=1.5)
    axes.set_title(title)
    axes.set_xticks(range(0, epochs, epochs // 10))
    axes.legend([metrics, val_met], loc='best')
    axes.set_ylabel(metrics)
    axes.grid()
    
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(20, 10))
fig.set_size_inches(14., 7.)
plot_ax(ax[0][0], 'loss', 'Loss', 200)
plot_ax(ax[0][1], 'accuracy', 'Akurasi', 200)
plot_ax(ax[1][0], 'prec', 'Presisi', 200)
plot_ax(ax[1][1], 'rec', 'Recall', 200)
fig = plt.gcf()
plt.suptitle('Grafik Setiap Metrics (CNN)\n(sumbu x adalah epoch)')
plt.show()

model.save('/content/drive/My Drive/Kompetisi/SATRIA DATA 2020/SEC/baseline.h5')

## Transfer Learning
from tensorflow.keras.applications import MobileNetV2, InceptionV3, VGG16, VGG19, ResNet50, InceptionResNetV2
# MobileNetV2
mobile_model=MobileNetV2(input_shape=(300,300,3),include_top=False)
mobile_model.trainable=False
mobile_model.summary()

last_output=mobile_model.output
print('last layer output shape: ', mobile_model.output_shape)

from tensorflow.keras import layers
from tensorflow.keras import Model

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(mobile_model.input, x)
model.summary

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy',prec,rec])

# Fit the model
history = model.fit(train_generator,
                              epochs=100,
                              verbose=1,
                              validation_data=validation_generator)

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(20, 10))
fig.set_size_inches(14., 7.)
plot_ax(ax[0][0], 'loss', 'Loss', 200)
plot_ax(ax[0][1], 'accuracy', 'Akurasi', 200)
plot_ax(ax[1][0], 'prec', 'Presisi', 200)
plot_ax(ax[1][1], 'rec', 'Recall', 200)
fig = plt.gcf()
plt.suptitle('Grafik Setiap Metrics (MobileNetV2)\n(sumbu x adalah epoch)')
plt.show()

model.save('/content/drive/My Drive/Kompetisi/SATRIA DATA 2020/SEC/mobiletrans.h5')

# InceptionV3
incept_model=InceptionV3(input_shape=(300,300,3),include_top=False)
incept_model.trainable=False
incept_model.summary()

last_output=incept_model.output
print('last layer output shape: ', incept_model.output_shape)

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(incept_model.input, x)
model.summary

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy',prec,rec])

# Fit the model
history = model.fit(train_generator,
                              epochs=100,
                              verbose=1,
                              validation_data=validation_generator)

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(20, 10))
fig.set_size_inches(14., 7.)
plot_ax(ax[0][0], 'loss', 'Loss', 200)
plot_ax(ax[0][1], 'accuracy', 'Akurasi', 200)
plot_ax(ax[1][0], 'prec', 'Presisi', 200)
plot_ax(ax[1][1], 'rec', 'Recall', 200)
fig = plt.gcf()
plt.suptitle('Grafik Setiap Metrics (InceptionV3)\n(sumbu x adalah epoch)')
plt.show()

model.save('/content/drive/My Drive/Kompetisi/SATRIA DATA 2020/SEC/Inceptrans.h5')

# VGG19
vgg19_model=VGG19(input_shape=(300,300,3),include_top=False)
vgg19_model.trainable=False
vgg19_model.summary()

last_output=vgg19_model.output
print('last layer output shape: ', vgg19_model.output_shape)

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(vgg19_model.input, x)
model.summary

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy',prec,rec])

# Fit the model
history = model.fit(train_generator,
                              epochs=100,
                              verbose=1,
                              validation_data=validation_generator)

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(20, 10))
fig.set_size_inches(14., 7.)
plot_ax(ax[0][0], 'loss', 'Loss', 200)
plot_ax(ax[0][1], 'accuracy', 'Akurasi', 200)
plot_ax(ax[1][0], 'prec', 'Presisi', 200)
plot_ax(ax[1][1], 'rec', 'Recall', 200)
fig = plt.gcf()
plt.suptitle('Grafik Setiap Metrics (VGG19)\n(sumbu x adalah epoch)')
plt.show()

model.save('/content/drive/My Drive/Kompetisi/SATRIA DATA 2020/SEC/VGG19trans.h5')
