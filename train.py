import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Create model folder
os.makedirs("model", exist_ok=True)

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15

train_path = r"C:\Users\guruprasad\Desktop\BrainTumor\dataset\train"
test_path = r"C:\Users\guruprasad\Desktop\BrainTumor\dataset\test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 tumor classes
])

model.compile(optimizer=Adam(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

model.save(r"C:\Users\guruprasad\Desktop\MRI\models\brain_tumor_model.h5")
print("Model saved successfully!")
