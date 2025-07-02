import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# === LOAD DATASET ===
df = pd.read_csv("HAM10000_metadata.csv")
df['path'] = df['image_id'].apply(lambda x: f"Dataset/{x}.jpg")
df['label'] = df['dx']

# === TRAIN/VAL SPLIT ===
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# === IMAGE GENERATORS ===
img_size = (224, 224)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size
)

# === RESNET50 MODEL ===
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
output = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === COMPILE & TRAIN ===
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

# === SAVE MODEL ===
model.save('resnet50_skin_cancer.keras')
