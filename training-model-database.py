import json
import csv
import pickle
import psycopg2
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split

from keras.src.applications.efficientnet_v2 import EfficientNetV2B1, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==================================================
# Configurações do banco e dataset
# ==================================================
DB_CONFIG = {
    "dbname": "image-classification",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

# Labels conhecidos (ordem fixa)
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
label_to_index = {label: idx for idx, label in enumerate(CLASS_NAMES)}

# ==================================================
# Caminhos de saída
# ==================================================
MODEL_PATH = 'modelo_flores.keras'
HISTORY_PATH_JSON = 'historico_treinamento.json'

# ==================================================
# Hiperparâmetros
# ==================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.0001
CHUNK_SIZE = 10000  # quantas imagens carregar do banco por vez

# ==================================================
# Função para carregar imagens do banco
# ==================================================
def load_images_from_db(limit=10000, offset=0):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT image, label FROM photos ORDER BY id LIMIT %s OFFSET %s", (limit, offset))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    X, y = [], []
    for img_bytes, label in rows:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        X.append(np.array(image))
        y.append(label_to_index[label])

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")

    return X, y

# ==================================================
# Criação do modelo com EfficientNetV2B1
# ==================================================
def create_model(num_classes=len(CLASS_NAMES)):
    base_model = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================================================
# Data augmentation (igual antes)
# ==================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ==================================================
# Treinamento em blocos vindos do banco
# ==================================================
model = create_model()
offset = 0
history_all = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
]

while True:
    X, y = load_images_from_db(limit=CHUNK_SIZE, offset=offset)
    if len(X) == 0:
        break

    print(f"Treinando com bloco iniciado no offset {offset}, {len(X)} imagens")

    # Split treino/validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # Geradores (augment no treino, apenas preprocess no val)
    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    # Treinamento do modelo nesse bloco
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Acumula histórico
    for k in history_all.keys():
        history_all[k].extend(history.history[k])

    offset += CHUNK_SIZE

# ==================================================
# Salvando o histórico
# ==================================================
with open(HISTORY_PATH_JSON, 'w') as f_json:
    json.dump(history_all, f_json)

print("Treinamento concluído e histórico salvo.")
