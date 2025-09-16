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

# Escolha da classe para classificação binária
TARGET_CLASS = "roses"

# ==================================================
# Caminhos de saída
# ==================================================
MODEL_PATH = 'modelo_rosa.keras'
HISTORY_PATH_JSON = 'historico_treinamento_rosa.json'

# ==================================================
# Hiperparâmetros
# ==================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
CHUNK_SIZE = 10000  # quantas imagens carregar do banco por vez

# ==================================================
# Função para carregar imagens do banco
# ==================================================
def load_images_from_db(limit=10000, offset=0, target_size=(224, 224)):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT image, label FROM photos ORDER BY id LIMIT %s OFFSET %s", (limit, offset))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    X, y = [], []
    for img_bytes, label in rows:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        image = image.resize(target_size)
        X.append(np.array(image))

        # Para classificação binária: 1 se for TARGET_CLASS, 0 caso contrário
        y.append(1 if label == TARGET_CLASS else 0)

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")

    return X, y

# ==================================================
# Criação do modelo com EfficientNetV2B1 (binário)
# ==================================================
def create_model(input_shape=(224, 224, 3)):
    # Carrega a base EfficientNet pré-treinada sem a camada de classificação
    base_model = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congela os pesos da base

    # Extrai as features da base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Reduz HxW para vetor único por canal
    x = Dropout(0.5)(x)               # Dropout para reduzir overfitting
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Camada final para classificação binária
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compila o modelo com binary_crossentropy
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================================================
# Data augmentation
# ==================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ==================================================
# Treinamento em blocos
# ==================================================
model = create_model()
offset = 0
history_all = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
]

while True:
    # Carrega bloco de imagens do banco
    X, y = load_images_from_db(limit=CHUNK_SIZE, offset=offset)
    if len(X) == 0:
        break

    print(f"Treinando com bloco iniciado no offset {offset}, {len(X)} imagens")

    # Divide em treino e validação (80%/20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # Geradores para aplicar augmentation apenas no treino
    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    # Treina o modelo nesse bloco
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
# Salva histórico
# ==================================================
with open(HISTORY_PATH_JSON, 'w') as f_json:
    json.dump(history_all, f_json)

print("Treinamento binário concluído e histórico salvo.")
