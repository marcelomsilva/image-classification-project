# evaluate_from_db.py
# Avalia um modelo salvo usando imagens de VALIDAÇÃO vindas do banco de dados (Postgres)
# Em vez de usar flow_from_directory.
# Gera: gráficos de histórico, matriz de confusão e relatório de classificação.

import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import psycopg2
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# IMPORT do preprocess_input compatível com EfficientNetV2B1 (mesmo que usou no treino)
from keras.src.applications.efficientnet_v2 import preprocess_input

# ========================
# Configurações (ajuste conforme seu ambiente)
# ========================
history_path = "historico_treinamento.json"   # arquivo com histórico salvo
model_path = "modelo_flores.keras"            # modelo salvo
IMG_SIZE = (224, 224)                         # tamanho de entrada usado no treino
BATCH_SIZE = 32                               # quantas imagens processar por vez na avaliação

DB_CONFIG = {
    "dbname": "image-classification",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

# Lista de classes conhecida e fixa (mesma que você usou no treino)
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
label_to_index = {label: idx for idx, label in enumerate(CLASS_NAMES)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# ========================
# ETAPA 0 - carregar histórico e plotar (igual ao seu script)
# ========================
# Carrega o histórico salvo durante o treino (JSON)
with open(history_path, "r") as f:
    history = json.load(f)

# Plota acurácia e loss (train vs val)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.get("accuracy", []), label="Treino")
plt.plot(history.get("val_accuracy", []), label="Validação")
plt.title("Acurácia por Época")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.get("loss", []), label="Treino")
plt.plot(history.get("val_loss", []), label="Validação")
plt.title("Loss por Época")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("grafico_historico.png")
plt.show()

# ========================
# ETAPA 1 - Carregar o modelo treinado
# ========================
model = tf.keras.models.load_model(model_path)
# imprime resumo para checar shapes e número de parâmetros (útil para depuração)
model.summary()

# ========================
# ETAPA 2 - Pegar todos os IDs e criar split treino/val (vamos usar só val aqui)
# ========================
# Conectar e buscar todos os ids e labels do banco
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT id, label FROM photos ORDER BY id")
rows = cur.fetchall()  # lista de tuplas (id, label_text)
cur.close()
conn.close()

# Se não houver dados, aborta
if len(rows) == 0:
    raise SystemExit("Tabela 'photos' está vazia. Nada para avaliar.")

# Separa ids e labels (texto)
all_ids = [r[0] for r in rows]
all_labels_text = [r[1] for r in rows]

# Faz a divisão global 80% treino / 20% validação (estratificada, para manter proporção de classes)
# Usamos random_state fixo para reprodutibilidade
train_ids, val_ids, train_labels_text, val_labels_text = train_test_split(
    all_ids, all_labels_text, test_size=0.2, stratify=all_labels_text, random_state=42
)

# Para avaliação iremos usar val_ids / val_labels_text
print(f"Total imagens no banco: {len(all_ids)}")
print(f"Imagens de validação (usadas aqui): {len(val_ids)}")

# ========================
# ETAPA 3 - Função auxiliar: carregar imagens por uma lista de ids (mantendo ordem)
# ========================
def load_images_by_ids(conn_params, ids_list, target_size=(224, 224)):
    """
    Dado um conjunto de IDs (ids_list), busca (id, image, label) do banco,
    garante a ordem conforme ids_list e retorna:
      - X: np.array shape (N, H, W, 3) (uint8 -> mas retornamos float32 e sem preprocess)
      - y_text: lista de labels em texto na mesma ordem
    Observação: a query usa WHERE id = ANY(%s) e depois reordena manualmente.
    """
    if len(ids_list) == 0:
        return np.array([]), []

    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    # Busca os registros para esses ids (pode retornar em ordem diferente)
    cur.execute("SELECT id, image, label FROM photos WHERE id = ANY(%s)", (list(ids_list),))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Mapeia id -> (img_bytes, label_text)
    id_to_row = {row[0]: (row[1], row[2]) for row in rows}

    images = []
    labels_text = []

    # Garante a ordem original de ids_list
    for id_ in ids_list:
        if id_ not in id_to_row:
            # caso algum id não exista mais no banco, pula e emite aviso
            print(f"Warning: id {id_} não encontrado no banco; pulando.")
            continue
        img_bytes, label_text = id_to_row[id_]

        # Bytes -> PIL -> redimensiona -> np.array
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize(target_size)
        arr = np.array(img)  # uint8 (0..255)
        images.append(arr)
        labels_text.append(label_text)

    if len(images) == 0:
        return np.array([]), []

    X = np.array(images, dtype="float32")
    return X, labels_text

# ========================
# ETAPA 4 - Prever em batches sobre o conjunto de validação
# ========================
y_true_indices = []  # labels verdadeiros (inteiros)
y_pred_indices = []  # predições (inteiros)

# Processa em batches para economizar memória
for start in range(0, len(val_ids), BATCH_SIZE):
    batch_ids = val_ids[start:start + BATCH_SIZE]
    # Carrega imagens e labels textuais para esse batch (na ordem de batch_ids)
    X_batch, labels_text_batch = load_images_by_ids(DB_CONFIG, batch_ids, target_size=IMG_SIZE)

    if X_batch.size == 0:
        continue  # nada nesse batch (pulo por segurança)

    # Aplica preprocess_input do EfficientNetV2 (mesmo que usou no treino)
    # preprocess_input espera arrays tipo float; faz transformações específicas (normalização / scale)
    X_batch_proc = preprocess_input(X_batch)

    # Faz predição com o modelo para este batch
    preds = model.predict(X_batch_proc, verbose=0)  # shape (batch_size, num_classes)

    # Converte probabilidades para índices de classes
    preds_idx = np.argmax(preds, axis=1)

    # Converte labels textuais do banco para índices (usando CLASS_NAMES)
    true_idx = [label_to_index[label_text] for label_text in labels_text_batch]

    # Acumula
    y_true_indices.extend(true_idx)
    y_pred_indices.extend(preds_idx.tolist())

# Transforma em numpy arrays
y_true = np.array(y_true_indices, dtype=int)
y_pred = np.array(y_pred_indices, dtype=int)

print(f"Total avaliações realizadas: {len(y_true)}")

# ========================
# ETAPA 5 - Matriz de Confusão e Relatório
# ========================
# Gera matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
print("Matriz de Confusão (array):")
print(cm)

# Exibe matriz de confusão com rótulos das classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.title("Matriz de Confusão - Validação (do DB)")
plt.savefig("matriz_confusao_db.png")
plt.show()

# Relatório de classificação (precision, recall, f1-score por classe)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("===== RELATÓRIO DE CLASSIFICAÇÃO =====")
print(report)

# ========================
# ETAPA 6 - Se quiser, calcular métricas agregadas (accuracy geral)
# ========================
accuracy = np.mean(y_true == y_pred)
print(f"Acurácia no conjunto de validação (do DB): {accuracy:.4f}")

# Fim do script
