import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

def transform(datagen,train_labels,train_images,n = 10):
    counter= 0
    size = len(train_images)
    for i in range(size):
      img = train_images[i]
      print(counter/size)
      # Load Image
      img = np.asarray(img)
      img = np.expand_dims(img, axis=0)
    
      # Iterator
      aug_iter = datagen.flow(img, batch_size=1)
    
      # Generate batch of images
      for i in range(n):
    
        # Convert to unsigned integers
        image = next(aug_iter)[0].astype('uint8')
      
        # Plot imageimage)
        train_labels.append(train_labels[counter])
        train_images.append(image)
      counter = counter+1
      
def data_rotation(train_labels,train_images,n = 10):
    max_rot = 60
    
    datagen = ImageDataGenerator(rotation_range=max_rot, fill_mode='nearest')
    
    transform(datagen,train_labels,train_images)
    

def data_brightness(train_labels,train_images, n = 10):
    datagen = ImageDataGenerator(brightness_range=[0.4,1.9], fill_mode='nearest')
    
    transform(datagen,train_labels,train_images)

def data_shifts(train_labels,train_images, n = 10):
    datagen = ImageDataGenerator(width_shift_range=0.45, height_shift_range=0.45)
    
    transform(datagen,train_labels,train_images)

def data_flip(train_labels,train_images, n = 10):
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    
    transform(datagen,train_labels,train_images)


def data_zoom(train_labels,train_images, n = 10, with_plot=False):
    datagen = ImageDataGenerator(zoom_range=[0.3,1.2])
    
    transform(datagen,train_labels,train_images)
  
# https://www.tensorflow.org/tutorials/images/classification

dir_root = "./../data"

train_root = os.path.join(dir_root,"train")
test_root = os.path.join(dir_root,"test")

train = tf.keras.utils.image_dataset_from_directory(
    train_root,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=None,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

test = tf.keras.utils.image_dataset_from_directory(
    test_root,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=None,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

#look at data
import matplotlib.pyplot as plt


class_names = train.class_names
print(class_names)
if False:
    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        

train_x = []
train_y = []

test_x = []
test_y = []

for images, labels in train.take(len(train)):
      train_x.append(tf.get_static_value(images))
      train_y.append(tf.get_static_value(labels))


for images, labels in train.take(len(test)):
      test_x.append(tf.get_static_value(images))
      test_y.append(tf.get_static_value(labels))
        
#data augmentation
print(3+'s')
data_rotation(train_y,train_x,2)

data_brightness(train_y,train_x,2)
data_shifts(train_y,train_x,2)
data_flip(train_y,train_x,2)
data_zoom(train_y,train_x,2)

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#build model
num_classes = len(class_names)

img_height = 256
img_width = 256



model =models.Sequential([
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(32, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=100
history = model.fit(
  train,
  # validation_data=val_ds,
  epochs=epochs
)

#evaluate
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("Evaluate on test data")
results = model.evaluate(test)
print("test loss, test acc:", results)
