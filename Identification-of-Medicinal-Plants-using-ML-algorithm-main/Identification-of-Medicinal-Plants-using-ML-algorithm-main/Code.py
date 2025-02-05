import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL
  # Updated import

# Part 1 - Data Preprocessing
### Preprocessing the Training set

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
training_set = train_datagen.flow_from_directory('C:/Users/dharn/Plant detection/plant_detection/data_set/Train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'  # Use 'categorical' for multiple classes
)

### Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'C:/Users/dharn/Plant detection/plant_detection/data_set/Test',  # Make sure this path is correct
    target_size=(128, 128),
    batch_size=34,
    class_mode='categorical'  # Use 'categorical' for multiple classes
)

# Part 2 - Building the CNN
### Initialising the CNN
cnn = tf.keras.models.Sequential()

#### Step 1 – Convolution
cnn.add(tf.keras.layers.Conv2D(filters=50, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))

#### Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=60, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

#### Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#### Step 5 - Output Layer
num_plant_classes = 4  # Change this to the number of plant classes
cnn.add(tf.keras.layers.Dense(units=num_plant_classes, activation='softmax'))

### Part 3 - Training the CNN
### Compiling the CNN
from tensorflow.keras.optimizers import Adam  # Updated import
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

### Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

### Part 4 - Making a single prediction
from tensorflow.keras.preprocessing import image  # Updated import

# Loading an image you want to classify
test_image = image.load_img(r'C:\Users\dharn\Plant detection\plant_detection\data_set\Single_predection\tulsi.jpg', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0
result = cnn.predict(test_image)
plant_classes = training_set.class_indices
predicted_class = list(plant_classes.keys())[np.argmax(result)]

print("The predicted class is:", predicted_class)

# Saving the model
cnn.save('plant_model.h5')

# Loading the model
from tensorflow.keras.models import load_model  # Updated import
loaded_model = load_model('plant_model.h5')