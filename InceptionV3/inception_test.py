import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# --- Load validation generator again ---
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_DIR = 'Dataset'
CSV_PATH = 'HAM10000_metadata.csv'
IMG_SIZE = 224
BATCH_SIZE = 32

df = pd.read_csv(CSV_PATH)
df['filename'] = df['image_id'] + '.jpg'
df = df.rename(columns={'dx': 'label'})
df = df[df['filename'].isin(os.listdir(IMG_DIR))]

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMG_DIR,
    x_col='filename',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- Load trained model ---
model = load_model('inceptionv3_skin_cancer.h5')

# --- Predict ---
pred_probs = model.predict(val_gen)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# --- Classification Report ---
print("\n--- Classification Report ---")
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print(report)

# --- Confusion Matrix ---
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
