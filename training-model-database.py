# ==================================================
# Imports: bibliotecas necessárias e razões
# ==================================================
import json                    # salvar histórico em JSON (fácil leitura e interoperabilidade)
import psycopg2                # cliente PostgreSQL para buscar imagens do banco
import numpy as np             # manipulação de arrays numéricos (imagens como arrays)
from io import BytesIO         # para transformar bytes do DB em arquivo em memória
from PIL import Image          # abrir/transformar imagens a partir de bytes
from sklearn.model_selection import train_test_split  # dividir em treino/val (80/20)

# EfficientNetV2 (backbone pré-treinado) e preprocessamento específico
from keras.src.applications.efficientnet_v2 import EfficientNetV2B1, preprocess_input

# Keras: criar modelo, camadas, otimização e callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ImageDataGenerator: aplica data augmentation (aumenta diversidade dos dados de treino)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==================================================
# Configurações do banco e dataset
# ==================================================
# Parâmetros de conexão com o PostgreSQL. Mantenha estes dados seguros (variáveis de ambiente em produção).
DB_CONFIG = {
    "dbname": "image-classification",  # nome do banco
    "user": "postgres",                # usuário do banco
    "password": "postgres",            # senha do banco
    "host": "localhost",               # host do banco
    "port": 5432,                      # porta do PostgreSQL
}

# Lista de classes (labels) conhecida e fixa.
# Ter a lista fixa garante ordem consistente entre treinos/execuções.
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# Mapeamento texto -> inteiro (ex: "roses" -> 2). O modelo treina com inteiros (sparse labels).
# Escolhemos esse mapeamento único para manter consistência (mesma ordem de classes sempre).
label_to_index = {label: idx for idx, label in enumerate(CLASS_NAMES)}

# ==================================================
# Caminhos para salvar modelo e histórico
# ==================================================
MODEL_PATH = 'modelo_flores.keras'                 # arquivo final do modelo (formato .keras é recomendado)
HISTORY_PATH_JSON = 'historico_treinamento.json'   # histórico em JSON (legível)

# ==================================================
# Hiperparâmetros principais (ajustáveis)
# ==================================================
IMG_SIZE = 224           # altura/width; EfficientNetV2B1 geralmente espera 224x224
BATCH_SIZE = 160          # quantas amostras o modelo processa por passo (afeta memória e estabilidade do gradiente)
EPOCHS = 10              # máximo de épocas por bloco (callbacks podem parar antes)
LEARNING_RATE = 0.0001   # taxa de aprendizado inicial para o otimizador Adam
CHUNK_SIZE = 10000       # quantas imagens buscar do banco por vez (trade-off: menos queries vs. mais RAM)

# ==================================================
# Função para carregar imagens do banco em blocos
# ==================================================
def load_images_from_db(limit=10000, offset=0, target_size=(224, 224)):
    """
    Busca 'limit' imagens a partir do 'offset' da tabela photos.
    Retorna:
      X: np.array de shape (N, H, W, 3) com dtype float32 (imagens não normalizadas ainda)
      y: np.array de shape (N,) com dtype int32 contendo índices das classes
    Observações:
      - Usamos ORDER BY id + LIMIT/OFFSET por simplicidade. Para tabelas MUITO grandes,
        considere paginação por chave (keyset pagination) para melhor performance.
    """
    # Abre conexão com o banco para esta consulta específica
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Consulta: pega a imagem (BYTEA) e o label (texto), ordena por id para consistência entre blocos
    # OFFSET/LIMIT permitem paginação simples (pegar 0..9999, 10000..19999, ...)
    cur.execute("SELECT image, label FROM photos ORDER BY id LIMIT %s OFFSET %s", (limit, offset))

    # traz todas as linhas retornadas: lista de tuplas (img_bytes, label_text)
    rows = cur.fetchall()

    # fechamos cursor e conexão o quanto antes para liberar recursos
    cur.close()
    conn.close()

    # listas temporárias para armazenar imagens e labels convertidos
    X, y = [], []

    # iteramos sobre os resultados e convertemos cada imagem
    for img_bytes, label in rows:
        # Bytes -> arquivo em memória -> PIL Image
        # BytesIO cria um buffer em memória a partir dos bytes (não salva em disco)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        # Redimensiona para target_size (padronizar entradas ao modelo)
        image = image.resize(target_size)
        # Converte para numpy array (uint8) e adiciona à lista
        X.append(np.array(image))
        # Converte label textual para índice inteiro com label_to_index
        # assumimos que o label existe na lista CLASS_NAMES (se não, KeyError - você pode adicionar validação)
        y.append(label_to_index[label])

    # transforma listas em arrays numpy com dtypes apropriados
    X = np.array(X, dtype="float32")  # imagens em 0..255 (pré-processamento será aplicado depois)
    y = np.array(y, dtype="int32")    # labels como inteiros

    return X, y

# ==================================================
# Função que cria o modelo baseado em EfficientNetV2B1
# ==================================================
def create_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES)):
    """
    Cria e compila o modelo:
      - backbone EfficientNetV2B1 pré-treinado (pesos ImageNet)
      - topo customizado com GlobalAveragePooling + Dense + Dropout
      - compila com sparse_categorical_crossentropy (labels inteiros)
    """
    # Carrega a base pré-treinada sem a "top" (sem o classificador final)
    base_model = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congela a base: inicialmente não atualizamos esses pesos (economiza tempo e evita overfitting)
    base_model.trainable = False

    # pega a saída da base e aplica pooling global para reduzir o mapa de características a um vetor
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Dropout para reduzir overfitting (desliga aleatoriamente fração das unidades durante treino)
    x = Dropout(0.5)(x)

    # camada densa intermediária (capacidade de aprender combinação das features do backbone)
    x = Dense(128, activation='relu')(x)

    # mais dropout para regularização da camada densa
    x = Dropout(0.3)(x)

    # camada final: num_classes saídas com softmax para probabilidade por classe
    predictions = Dense(num_classes, activation='softmax')(x)

    # monta o modelo completo (entrada da base -> saída das camadas customizadas)
    model = Model(inputs=base_model.input, outputs=predictions)

    # compila o modelo
    # - Adam: bom otimizador por padrão
    # - sparse_categorical_crossentropy: usamos porque entregamos labels como inteiros (não one-hot)
    # - metrics=['accuracy']: avaliar acurácia durante treino/val
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ==================================================
# Data augmentation: aumenta diversidade das imagens de treino
# ==================================================
# train_datagen aplica transformações aleatórias nas imagens de treino (aumenta robustez)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # função de preprocessamento específica do EfficientNetV2
    rotation_range=20,    # rotações aleatórias até 20 graus
    zoom_range=0.2,       # zoom aleatório até +/-20%
    horizontal_flip=True  # flip horizontal aleatório
)

# val_datagen apenas aplica o preprocess (sem augment), para validação consistente
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ==================================================
# Treinamento em blocos vindo do banco
# ==================================================
# Observação: removi o uso de "if __name__ == '__main__':" porque você pediu que não fosse necessário.
model = create_model()  # cria e compila o modelo uma vez

# offset usado para paginação simples no banco (0, CHUNK_SIZE, 2*CHUNK_SIZE, ...)
offset = 0

# dicionário para acumular histórico de treinamento de todos os blocos
# usamos listas para poder estender com cada history retornado por model.fit
history_all = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

# Callbacks:
# - EarlyStopping: interrompe se val_loss não melhorar por 'patience' épocas (evita treinar demais)
# - ModelCheckpoint: salva o melhor modelo encontrado com base em 'val_loss'
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
]

# Loop principal: carrega blocos do banco e treina o modelo até esgotar as imagens
while True:
    # carrega um bloco de CHUNK_SIZE imagens a partir do offset atual
    X, y = load_images_from_db(limit=CHUNK_SIZE, offset=offset)

    # Se não houver mais imagens no banco (retornou lista vazia), finaliza o loop
    if len(X) == 0:
        break

    # mensagem informativa para acompanhamento (útil em logs)
    print(f"Treinando com bloco iniciado no offset {offset}, {len(X)} imagens")

    # Divisão treino/validação dentro do bloco (80% / 20%)
    # stratify=y garante que a proporção de classes seja mantida em ambos conjuntos
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # Criamos geradores a partir dos arrays:
    # - train_gen aplica augment e o preprocess_input via ImageDataGenerator
    # - val_gen apenas aplica preprocess_input (sem augment)
    # flow aplica as transformações e gera batches para o model.fit
    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    # Treina o modelo com os dados do bloco atual.
    # Observações:
    # - o modelo já pode ter sido treinado em blocos anteriores; aqui continuamos o treinamento incrementalmente.
    # - callbacks controlam salvamento e parada precoce.
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # history.history contém arrays por época para as métricas (ex: loss, accuracy, val_loss, val_accuracy)
    # estendemos history_all com os resultados deste bloco
    for k in history_all.keys():
        # extend adiciona cada elemento do history deste bloco (1..n épocas) à lista acumulada
        history_all[k].extend(history.history[k])

    # incrementa o offset para o próximo bloco do banco
    offset += CHUNK_SIZE

# ==================================================
# Após concluir todos os blocos: salvar histórico em múltiplos formatos
# ==================================================
# JSON: formato legível (bom para análise externa)
with open(HISTORY_PATH_JSON, 'w') as f_json:
    json.dump(history_all, f_json)

# Mensagem final
print("Treinamento concluído e histórico salvo.")
