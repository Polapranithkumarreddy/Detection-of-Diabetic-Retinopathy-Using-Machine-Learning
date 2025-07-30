import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Paths ===
data_dir = "diabetic detection using retinopathy//gaussian_filtered_images"
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'dr_model.h5')

# === Classes ===
class_names = ['No_DR', 'Mild', 'Moderate', 'ERRORDATA']

# === Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=class_names
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=class_names
)

# === Class Weights ===
if train_generator.samples == 0:
    raise ValueError("No training data found.")
if val_generator.samples == 0:
    raise ValueError("No validation data found.")

print("Class indices:", train_generator.class_indices)
print("Train class distribution:", np.bincount(train_generator.classes))

classes = np.array(list(train_generator.class_indices.values()))
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_generator.classes
)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# === Load Pretrained VGG16 (without top) ===
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

# === Freeze base_model layers ===
for layer in base_model.layers:
    layer.trainable = False

# === Add Custom Head ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Compile ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# === Train ===
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# === Evaluate QWK ===
val_generator.reset()
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
kappa = cohen_kappa_score(y_true, y_pred_classes, weights='quadratic')
print(f"Quadratic Weighted Kappa: {kappa:.4f}")

# === Save Model ===
model.save(model_path)
print(f"Model saved to {model_path}")
