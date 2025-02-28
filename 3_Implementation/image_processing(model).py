import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ======================
# Dataset Configuration
# ======================
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
SEED = 42
BASE_DIR = '/content/drive/MyDrive/sem-4AIML/soil_detection/organized_dataset'

# ======================
# Data Loading & Preprocessing
# ======================
# Load training set first to get class names
train_ds = image_dataset_from_directory(
    f'{BASE_DIR}/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode='int'
)
class_names = train_ds.class_names

# Load validation and test sets with consistent class order
val_ds = image_dataset_from_directory(
    f'{BASE_DIR}/validation',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False,
    label_mode='int'
)

test_ds = image_dataset_from_directory(
    f'{BASE_DIR}/test',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False,
    label_mode='int'
)

# Calculate class weights
def get_class_weights(dataset):
    class_counts = Counter(np.concatenate([y.numpy() for _, y in dataset]))
    total = sum(class_counts.values())
    return {cls: total/(len(class_counts)*count) for cls, count in class_counts.items()}

class_weights = get_class_weights(train_ds)
print("Class weights:", class_weights)

# ======================
# Data Augmentation & Preprocessing
# ======================
# Enhanced augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.GaussianNoise(0.01),
])

# EfficientNet preprocessing
def preprocess(image, label):
    image = applications.efficientnet.preprocess_input(image)
    return image, label

# Prepare datasets
def prepare(ds, augment=False):
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                   num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

# ======================
# Model Architecture
# ======================
def build_model():
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    # Freeze base model initially
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

model = build_model()

# ======================
# Training Configuration
# ======================
# Custom cosine decay with warmup
class WarmupCosineDecay(callbacks.Callback):
    def __init__(self, total_steps, warmup_steps, initial_lr):
        super().__init__()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr

    def on_train_begin(self, logs=None):
        self.step = 0

    def on_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step < self.warmup_steps:
            lr = self.initial_lr * (self.step/self.warmup_steps)
        else:
            progress = (self.step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.initial_lr * (1 + np.cos(np.pi * progress))

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

# Callbacks
callbacks = [
    callbacks.ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]

# ======================
# Training Process
# ======================
# Initial training
model.compile(
    optimizer=optimizers.Adam(1e-3, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

initial_epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# Fine-tuning
model = tf.keras.models.load_model('best_model.keras')
model.trainable = True

# Unfreeze top 50 layers
for layer in model.layers[1].layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(1e-5, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 20
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=history.epoch[-1] + 1,
    epochs=history.epoch[-1] + fine_tune_epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# ======================
# Evaluation
# ======================
# Final evaluation
def evaluate_model(dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

print("Validation Set Evaluation:")
evaluate_model(val_ds)

print("\nTest Set Evaluation:")
evaluate_model(test_ds)

# ======================
# Visualization
# ======================
def plot_training_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()

plot_training_curves(history_fine)
