import os
import random
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input

base_path = r"c:\Users\Karji\OneDrive\Desktop\PROJECT\proj2"
df = pd.read_csv(os.path.join(base_path, "train.csv"))

samples = 2500
df = df.loc[:samples, :]

data = pd.DataFrame(df["landmark_id"].value_counts())
data.reset_index(inplace=True)
data.columns = ['index', 'landmark_id']

frequent_landmark_ids = set(data[data['landmark_id'] >= 1]['index'].tolist())
df_filtered = df[df['landmark_id'].isin(frequent_landmark_ids)].copy()

if 'id' in df.columns:
    id_col = 'id'
elif 'image_id' in df.columns:
    id_col = 'image_id'
else:
    id_col = [c for c in df.columns if c != 'landmark_id']
if isinstance(id_col, list):
    id_col = id_col[0]

def build_safe_path(x, base):
    x_str = str(x)
    if x_str.startswith("train/") or x_str.startswith("train\\") or x_str.startswith("proj2/") or x_str.startswith("proj2\\"):
        x_str = x_str.split("/", 1)[-1].split("\\", 1)[-1]
    full_path = os.path.join(base, x_str)
    if not full_path.lower().endswith(".jpg"):
        full_path += ".jpg"
    return full_path

df_filtered['filepath'] = df_filtered[id_col].apply(lambda x: build_safe_path(x, base_path))

df_filtered['exists'] = df_filtered['filepath'].apply(os.path.exists)
df_filtered = df_filtered[df_filtered['exists'] == True].copy()

df_filtered['label_str'] = df_filtered['landmark_id'].astype(str)
unique_labels = sorted(df_filtered['label_str'].unique())
label_to_idx = {name: i for i, name in enumerate(unique_labels)}
idx_to_label = {i: name for name, i in label_to_idx.items()}
df_filtered['label_idx'] = df_filtered['label_str'].map(label_to_idx)

num_classes_filtered = len(unique_labels)

df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
val_split = int(len(df_filtered) * 0.2)

train_df = df_filtered.iloc[val_split:]
val_df = df_filtered.iloc[:val_split]

del df
gc.collect()

@tf.function
def parse_image(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, (64, 64))
    image = image / 255.0
    return image, label

train_filenames = train_df['filepath'].values
train_labels = train_df['label_idx'].values
val_filenames = val_df['filepath'].values
val_labels = val_df['label_idx'].values

BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=200)
train_dataset = train_dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
val_dataset = val_dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

inputs = Input(shape=(64, 64, 3))
base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(num_classes_filtered, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTraining on {len(train_df)} samples, validating on {len(val_df)} samples.")
print(f"Total target classes: {num_classes_filtered}\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o', color='orange')
plt.legend()
plt.title('Accuracy History')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o', color='orange')
plt.legend()
plt.title('Loss History')
plt.grid(True)

for img_batch, label_batch in val_dataset.take(1):
    predictions_matrix = model.predict(img_batch, verbose=0)
    display_count = min(3, len(img_batch))
    random_indices = random.sample(range(len(img_batch)), display_count)
    
    for plot_idx, sample_idx in enumerate(random_indices):
        plt.subplot(2, 3, 4 + plot_idx)
        plt.imshow(img_batch[sample_idx].numpy())
        
        true_class_idx = label_batch[sample_idx].numpy()
        pred_class_idx = np.argmax(predictions_matrix[sample_idx])
        
        true_name = idx_to_label[true_class_idx]
        pred_name = idx_to_label[pred_class_idx]
        
        text_color = 'green' if true_class_idx == pred_class_idx else 'red'
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=text_color, fontsize=8)
        plt.axis('off')

plt.tight_layout()
plt.show(block=True)
