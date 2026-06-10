import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------- CONFIG ----------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

DATASET_DIR = "dataset"
CSV_FILE = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")

PART1 = os.path.join(DATASET_DIR, "HAM10000_images_part_1")
PART2 = os.path.join(DATASET_DIR, "HAM10000_images_part_2")

print("Loading dataset...")

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_FILE)

# ---------------- IMAGE PATH MAPPING ----------------
image_paths = {}

for folder in [PART1, PART2]:
    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            image_id = file.replace(".jpg", "")
            image_paths[image_id] = os.path.join(folder, file)

df["path"] = df["image_id"].map(image_paths)
df = df.dropna()

print("Images found:", len(df))

# ---------------- LABEL ENCODING ----------------
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["dx"])

np.save("class_names.npy", encoder.classes_)

# ---------------- LOAD IMAGES ----------------
X = []
y = []

for _, row in df.iterrows():
    try:
        img = Image.open(row["path"]).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img = np.array(img).astype("float32") / 255.0

        X.append(img)
        y.append(row["label"])

    except Exception:
        pass

X = np.array(X, dtype=np.float32)
y = tf.keras.utils.to_categorical(y)

print("Loaded Images:", len(X))

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

# ---------------- MODEL ----------------
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- TRAINING ----------------
print("Training Started...")

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ---------------- EVALUATION ----------------
loss, acc = model.evaluate(X_test, y_test)
print("Final Accuracy:", acc)

# ---------------- SAVE MODEL ----------------
os.makedirs("models", exist_ok=True)

model.save("models/skin_disease_model.h5")

print("Model Saved Successfully")
print("Path: models/skin_disease_model.h5")