import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
CSV_PATH = "HAM10000_metadata.csv"
IMG_FOLDER = "Dataset"
MODEL_PATH = "xception_skin_cancer.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# === 1. LOAD METADATA ===
df = pd.read_csv(CSV_PATH)
df['path'] = df['image_id'].apply(lambda x: f"{IMG_FOLDER}/{x}.jpg")
df['label'] = df['dx']

# === 2. SPLIT INTO TRAIN AND VALIDATION (just to get val_gen) ===
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# === 3. PREPARE VALIDATION GENERATOR ===
val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# === 4. LOAD TRAINED MODEL ===
model = load_model(MODEL_PATH)

# === 5. PREDICT ON VALIDATION SET ===
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = list(val_gen.class_indices.keys())

# === 6. PRINT CLASSIFICATION REPORT ===
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# === 7. CONFUSION MATRIX HEATMAP ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix - Xception Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("xception_confusion_matrix.png")
plt.show()
