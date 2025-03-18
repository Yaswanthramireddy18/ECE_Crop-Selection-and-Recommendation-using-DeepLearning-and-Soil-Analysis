import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, 
                                     Dropout, Input, Add, BatchNormalization, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

DATA_PATH = r"C:\New folder (3)\Dataset"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VAL_DIR = os.path.join(DATA_PATH, "validation")
TEST_DIR = os.path.join(DATA_PATH, "test")
MODEL_SAVE_PATH = "custom_cnn_best.h5"  

BATCH_SIZE = 64
IMG_SIZE = (128, 128)
EPOCHS = 50
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4  

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

test_gen = val_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)


def build_custom_cnn():
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # 🔹 Block 1
    x = Conv2D(32, (3, 3), activation="swish", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="swish", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    shortcut = Conv2D(64, (1, 1), strides=(2, 2), padding="same")(x)  # Downsampling
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="swish", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Add()([x, shortcut])  

    
    x = Conv2D(128, (3, 3), activation="swish", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation="swish", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    x = Flatten()(x)
    x = Dense(256, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(train_gen.num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

def lr_schedule(epoch, lr):
    return lr * 0.9  

optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)


model = build_custom_cnn()
model.compile(optimizer=optimizer, loss=focal_loss(), metrics=["accuracy"])


callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
    LearningRateScheduler(lr_schedule)  
]


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=callbacks,
    verbose=1  
)


model.load_weights(MODEL_SAVE_PATH)

tta_steps = 10
predictions = []
for _ in range(tta_steps):
    try:
        test_gen.reset()  
    except AttributeError:
        pass  
    
    preds = model.predict(test_gen, verbose=1)  
    predictions.append(preds)

final_predictions = np.mean(predictions, axis=0)
accuracy = np.mean(np.argmax(final_predictions, axis=1) == test_gen.classes)
print(f"\n🔥 Final Test Accuracy with TTA: {accuracy:.2%} 🔥")

model.save("custom_cnn_production_model.keras")
print(f"\n✅ Model saved at: custom_cnn_production_model.keras")

