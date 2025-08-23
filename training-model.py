import json
import csv
import pickle

from keras.src.applications.efficientnet_v2 import EfficientNetV2B3, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0  # Modelo pré-treinado
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Caminhos dos diretórios
DATASET_DIR = '/home/marcelo/Documents/estudos/projetos/flower_photos'        # Pasta onde estão as imagens separadas por classe
MODEL_PATH = 'modelo_flores.h5'  # Caminho onde o modelo treinado será salvo
HISTORY_PATH_PKL = 'historico_treinamento.pkl'  # Caminho para salvar o histórico de treinamento
HISTORY_PATH_CSV = 'historico_treinamento.csv'  # Caminho para salvar o histórico de treinamento
HISTORY_PATH_JSON = 'historico_treinamento.json'  # Caminho para salvar o histórico de treinamento

# Hiperparâmetros
IMG_SIZE = 224          # Tamanho da imagem (224x224 é o padrão para EfficientNetB0)
BATCH_SIZE = 32         # Número de imagens por lote durante o treinamento
EPOCHS = 40             # Quantidade de épocas para treinar
LEARNING_RATE = 0.0001  # Taxa de aprendizado

# Gerador de dados com aumentação de imagens para o treino e pré-processamento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Aplica o pré-processamento padrão do EfficientNet
    validation_split=0.2,                     # Separa 20% das imagens para validação
    rotation_range=20,                        # Rotaciona imagens até 20 graus
    zoom_range=0.2,                           # Aplica zoom aleatório de até 20%
    horizontal_flip=True                      # Espelha imagens horizontalmente
)

# Carrega o conjunto de treino com aumentação
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,                        # Diretório base com as subpastas por classe
    target_size=(IMG_SIZE, IMG_SIZE),  # Redimensiona as imagens
    batch_size=BATCH_SIZE,             # Tamanho dos lotes
    class_mode='categorical',          # Usa codificação one-hot para as classes
    subset='training'                  # Usa os 80% para treino
)

# Carrega o conjunto de validação (sem aumentação, mas com o mesmo pré-processamento)
val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'                # Usa os 20% restantes para validação
)

# Número de classes (baseado no número de subpastas do dataset)
num_classes = train_generator.num_classes

# Carrega a EfficientNetB0 pré-treinada (sem o topo/classificador final)
base_model = EfficientNetV2B3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Congela os pesos da base (não serão atualizados durante o treinamento)
base_model.trainable = False

# Adiciona camadas finais personalizadas no topo da base congelada
x = base_model.output                              # Saída da EfficientNet
x = GlobalAveragePooling2D()(x)                    # Reduz as dimensões finais em média por canal
x = Dropout(0.5)(x)                                 # Dropout para evitar overfitting
x = Dense(128, activation='relu')(x)               # Camada densa intermediária
x = Dropout(0.3)(x)                                 # Mais um dropout
predictions = Dense(num_classes, activation='softmax')(x)  # Camada final de classificação

# Cria o modelo final unindo a base com as novas camadas
model = Model(inputs=base_model.input, outputs=predictions)

# Compila o modelo com otimizador, função de perda e métrica de acurácia
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks:
# - EarlyStopping: interrompe o treinamento se a validação parar de melhorar
# - ModelCheckpoint: salva o melhor modelo baseado na acurácia de validação
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
]

# Treina o modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Salva no novo formato .keras (SavedModel)
model.save("modelo_flores.keras")

# Salva o histórico de treinamento em um arquivo .pkl (pickle)
with open(HISTORY_PATH_PKL, 'wb') as f:
    pickle.dump(history.history, f)

# JSON
with open(HISTORY_PATH_JSON, 'w') as f_json:
    json.dump(history.history, f_json)

# CSV
with open(HISTORY_PATH_CSV, 'w', newline='') as f_csv:
    writer = csv.writer(f_csv)
    header = ['epoch'] + list(history.history.keys())
    writer.writerow(header)
    for i in range(EPOCHS):
        row = [i + 1] + [history.history[k][i] for k in history.history.keys()]
        writer.writerow(row)
