import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. CRASH PREVENTION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# --- 2. MULTI-DIGIT DATASET GENERATOR ---
IMG_SIZE = 64

def generate_digit_dataset(num_samples=600, specific_labels=None):
    """Generates crisp digit arrays where boxes exactly map to pixel bounding edges."""
    images = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    bboxes = np.zeros((num_samples, 4), dtype=np.float32)
    
    # If specific labels are passed (for validation), use them. Otherwise, randomise 0-9.
    if specific_labels is not None:
        labels = np.array(specific_labels, dtype=np.int32)
        num_samples = len(specific_labels)
    else:
        labels = np.random.randint(0, 10, size=(num_samples,)) # Supports digits 0 through 9
    
    fig = plt.figure(figsize=(1, 1), dpi=IMG_SIZE)
    ax = fig.add_axes([0, 0, 1, 1]) # Spans entire canvas with zero padding margins
    
    for i in range(num_samples):
        ax.clear()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Consistent bounding parameters to maintain perfect structural framing
        xmin = np.random.randint(14, 18)
        ymin = np.random.randint(14, 18)
        box_width = 30
        box_height = 34
        
        xmax = xmin + box_width
        ymax = ymin + box_height
        
        # Render the text utilizing a sharp monospaced layout engine
        ax.text(xmin, IMG_SIZE - ymin, str(labels[i]), color='white', 
                fontsize=28, fontweight='bold', family='monospace',
                ha='left', va='top')
        
        ax.set_xlim(0, IMG_SIZE)
        ax.set_ylim(0, IMG_SIZE)
        ax.axis('off')
        
        # Extract pixel data from the figure canvas
        fig.canvas.draw()
        img_raw = np.asarray(fig.canvas.buffer_rgba())
        img_rgb = img_raw[:, :, :3]
        
        images[i, :, :, 0] = np.mean(img_rgb, axis=-1) / 255.0
        bboxes[i] = [ymin/IMG_SIZE, xmin/IMG_SIZE, ymax/IMG_SIZE, xmax/IMG_SIZE]
        
    plt.close(fig)
    return images, labels, bboxes

print("Generating diverse training digits (0-9)...")
x_train, y_train_labels, y_train_boxes = generate_digit_dataset(600)

# Force the 3 validation samples to be completely different numbers (e.g., 5, 2, and 8)
print("Generating 3 distinct validation digits...")
x_val, y_val_labels, y_val_boxes = generate_digit_dataset(specific_labels=[5, 2, 8])

# --- 3. EXPANDED CAPACITY CNN MODEL ---
def build_precision_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    
    c_dense = layers.Dense(128, activation='relu')(x)
    b_dense = layers.Dense(128, activation='relu')(x)
    
    # Outputting 10 nodes to support classification probabilities for digits 0 to 9
    class_out = layers.Dense(10, activation='softmax', name='class_output')(c_dense)
    box_out = layers.Dense(4, activation='sigmoid', name='box_output')(b_dense)
    
    return models.Model(inputs=inputs, outputs=[class_out, box_out])

model = build_precision_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'class_output': 'sparse_categorical_crossentropy', 'box_output': 'mse'},
    loss_weights={'class_output': 1.0, 'box_output': 10.0}
)

# --- 4. MODEL TRAINING ---
print("Training model across multiple digit structures...")
model.fit(
    x_train, 
    {'class_output': y_train_labels, 'box_output': y_train_boxes},
    epochs=15,       
    batch_size=32,   
    verbose=1
)

# --- 5. VISUALIZING DIVERSE SAMPLES ---
print("\nPlotting output window with unique digits...")
pred_classes, pred_boxes = model.predict(x_val)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Distinct Digits Output: Green (Ground Truth) | Red Dashed (Prediction)', 
             fontsize=12, fontweight='bold')

for i in range(3):
    ax = axes[i]
    ax.imshow(x_val[i].squeeze(), cmap='gray')
    
    t_ymin, t_xmin, t_ymax, t_xmax = y_val_boxes[i] * IMG_SIZE
    p_ymin, p_xmin, p_ymax, p_xmax = pred_boxes[i] * IMG_SIZE
    
    # Ground Truth Box
    rect_true = plt.Rectangle((t_xmin, t_ymin), t_xmax - t_xmin, t_ymax - t_ymin,
                              fill=False, color='green', linewidth=2.5)
    ax.add_patch(rect_true)
    
    # Prediction Box
    rect_pred = plt.Rectangle((p_xmin, p_ymin), p_xmax - p_xmin, p_ymax - p_ymin,
                             fill=False, color='red', linestyle='--', linewidth=2.5)
    ax.add_patch(rect_pred)
    
    actual_val = y_val_labels[i]
    predicted_val = np.argmax(pred_classes[i])
    
    ax.set_title(f"Sample #{i+1} (Digit: {actual_val})")
    ax.set_xlabel(f"True Digit: {actual_val}\nPredicted Digit: {predicted_val}", fontsize=10, color='cyan')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
print("Process completed successfully. Check the pop-up chart window.")
plt.show()
