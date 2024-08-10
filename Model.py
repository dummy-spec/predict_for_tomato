import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Path to your data
base_dir = os.path.join(os.path.dirname(__file__), '..', 'PlantVillage')

# Tomato-related folders
tomato_folders = [
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# Potato-related folders
potato_folders = [
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight'
]

# Pepper-related folders (just for reference, won't be used in this case)
pepper_folders = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy'
]

# Create paths
tomato_dir = os.path.join(base_dir, 'Tomato')
potato_dir = os.path.join(base_dir, 'Potato')

print(tomato_dir)

# Image data generators for Tomato
tomato_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # 20% for validation
)

tomato_train_generator = tomato_datagen.flow_from_directory(
    base_dir,
    classes=tomato_folders,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

tomato_validation_generator = tomato_datagen.flow_from_directory(
    base_dir,
    classes=tomato_folders,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# Image data generators for Potato
potato_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # 20% for validation
)

potato_train_generator = potato_datagen.flow_from_directory(
    base_dir,
    classes=potato_folders,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

potato_validation_generator = potato_datagen.flow_from_directory(
    base_dir,
    classes=potato_folders,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

def build_and_train_model(train_generator, validation_generator, num_classes, model_name):
    # Load the pre-trained MobileNetV2 model + higher layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the convolutional base
    base_model.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=10
    )

    # Save the model
    model.save(f'{model_name}.h5')

    # Plot training and validation accuracy/loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'{model_name} - Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss')

    plt.show()

# Train the Tomato model
build_and_train_model(tomato_train_generator, tomato_validation_generator, len(tomato_folders), 'Tomato_Disease_Model')

# Train the Potato model
build_and_train_model(potato_train_generator, potato_validation_generator, len(potato_folders), 'Potato_Disease_Model')
