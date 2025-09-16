# evaluate_from_db_binary.py
"""
Avalia um modelo binário (sigmoid) salvo usando imagens de VALIDAÇÃO vindas do banco de dados (Postgres).
Gera: gráfico de histórico, matriz de confusão e relatório de classificação.

Ajuste as configurações no topo conforme seu ambiente.
"""

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
history_path = "./historico_treinamento_rosa.json"   # arquivo com histórico salvo (ajuste se necessário)
model_path = "modelo_rosa.keras"                    # modelo salvo (ajuste se necessário)
IMG_SIZE = (224, 224)                               # tamanho de entrada usado no treino
BATCH_SIZE = 32                                     # quantas imagens processar por vez na avaliação

DB_CONFIG = {
    "dbname": "image-classification",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

# Classe alvo (string tal como armazenada em 'label' no banco)
TARGET_CLASS = "roses"

# Rótulos que serão mostrados no relatório/matriz (index 0 = negativo, index 1 = positivo)
NEG_LABEL = f"not_{TARGET_CLASS}"
POS_LABEL = TARGET_CLASS
BINARY_LABELS = [NEG_LABEL, POS_LABEL]

# ========================
# ETAPA 0 - carregar histórico e plotar (robusto se chaves não existirem)
# ========================
try:
    with open(history_path, "r") as f:
        history = json.load(f)
except FileNotFoundError:
    print(f"Warning: histórico não encontrado em '{history_path}'. Pulando plotting de histórico.")
    history = {}

# Plota acurácia e loss (train vs val) se as chaves existirem
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
if "accuracy" in history or "val_accuracy" in history:
    plt.plot(history.get("accuracy", []), label="Treino")
    plt.plot(history.get("val_accuracy", []), label="Validação")
    plt.title("Acurácia por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
else:
    plt.text(0.5, 0.5, "accuracy/val_accuracy não disponíveis no histórico", ha="center")
    plt.axis("off")

plt.subplot(1, 2, 2)
if "loss" in history or "val_loss" in history:
    plt.plot(history.get("loss", []), label="Treino")
    plt.plot(history.get("val_loss", []), label="Validação")
    plt.title("Loss por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
else:
    plt.text(0.5, 0.5, "loss/val_loss não disponíveis no histórico", ha="center")
    plt.axis("off")

plt.tight_layout()
plt.savefig("grafico_historico_binary.png")
plt.show()

# ========================
# ETAPA 1 - Carregar o modelo treinado
# ========================
model = tf.keras.models.load_model(model_path)
model.summary()

# ========================
# ETAPA 2 - Pegar todos os IDs e criar split treino/val (vamos usar só val aqui)
# ========================
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT id, label FROM photos ORDER BY id")
rows = cur.fetchall()  # lista de tuplas (id, label_text)
cur.close()
conn.close()

if len(rows) == 0:
    raise SystemExit("Tabela 'photos' está vazia. Nada para avaliar.")

all_ids = [r[0] for r in rows]
all_labels_text = [r[1] for r in rows]

# Convertemos os labels textuais para binários para stratify (1 se TARGET_CLASS, 0 caso contrário)
y_binary_all = [1 if lab == TARGET_CLASS else 0 for lab in all_labels_text]

# Faz split 80/20 estratificado (se possível)
stratify_param = y_binary_all if len(set(y_binary_all)) > 1 else None
train_ids, val_ids, train_labels_text, val_labels_text = train_test_split(
    all_ids, all_labels_text, test_size=0.2, stratify=stratify_param, random_state=42
)

print(f"Total imagens no banco: {len(all_ids)}")
print(f"Imagens de validação (usadas aqui): {len(val_ids)}")

# ========================
# ETAPA 3 - Função auxiliar: carregar imagens por uma lista de ids (mantendo ordem)
# ========================
def load_images_by_ids(conn_params, ids_list, target_size=(224, 224)):
    """
    Retorna (X, labels_text) na ordem de ids_list.
    X: np.array (N, H, W, 3) dtype float32 (valores 0..255 antes do preprocess_input)
    labels_text: lista de labels textuais
    """
    if len(ids_list) == 0:
        return np.array([]), []

    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.execute("SELECT id, image, label FROM photos WHERE id = ANY(%s)", (list(ids_list),))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    id_to_row = {row[0]: (row[1], row[2]) for row in rows}

    images = []
    labels_text = []

    for id_ in ids_list:
        if id_ not in id_to_row:
            print(f"Warning: id {id_} não encontrado no banco; pulando.")
            continue
        img_bytes, label_text = id_to_row[id_]

        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img = img.resize(target_size, Image.BILINEAR)
            arr = np.array(img, dtype="float32")
        except Exception as e:
            print(f"Warning: erro ao decodificar imagem id {id_}: {e}. Substituindo por zeros.")
            arr = np.zeros((*target_size, 3), dtype="float32")

        images.append(arr)
        labels_text.append(label_text)

    if len(images) == 0:
        return np.array([]), []

    X = np.array(images, dtype="float32")
    return X, labels_text

# ========================
# ETAPA 4 - Prever em batches sobre o conjunto de validação
# ========================
y_true = []
y_pred = []

for start in range(0, len(val_ids), BATCH_SIZE):
    batch_ids = val_ids[start:start + BATCH_SIZE]
    X_batch, labels_text_batch = load_images_by_ids(DB_CONFIG, batch_ids, target_size=IMG_SIZE)

    if X_batch.size == 0:
        continue

    # aplica preprocess_input (EfficientNet)
    X_batch_proc = preprocess_input(X_batch)

    # predição (modelo binário com saída sigmoide -> shape (N,1))
    preds = model.predict(X_batch_proc, verbose=0)  # shape (N,1) ou (N,)
    preds = np.asarray(preds).reshape(-1)  # vetor 1D de probabilidades

    # threshold 0.5 para converter em 0/1
    preds_bin = (preds >= 0.5).astype(int).tolist()

    # true labels binários
    true_bin = [1 if lt == TARGET_CLASS else 0 for lt in labels_text_batch]

    y_true.extend(true_bin)
    y_pred.extend(preds_bin)

y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)

print(f"Total avaliações realizadas: {len(y_true)}")

# ========================
# ETAPA 5 - Matriz de Confusão e Relatório (binário)
# ========================
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Matriz de Confusão (array):")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=BINARY_LABELS)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.title(f"Matriz de Confusão - Validação (binário: {TARGET_CLASS})")
plt.savefig("matriz_confusao_db_binary.png")
plt.show()

# Relatório (precision/recall/f1)
report = classification_report(y_true, y_pred, target_names=BINARY_LABELS, digits=4)
print("===== RELATÓRIO DE CLASSIFICAÇÃO (BINÁRIO) =====")
print(report)

# ========================
# ETAPA 6 - Métrica agregada
# ========================
accuracy = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0
print(f"Acurácia no conjunto de validação (do DB): {accuracy:.4f}")

# opcional: salvar resultados simples em JSON
results = {
    "total_eval": int(len(y_true)),
    "accuracy": float(accuracy),
    "confusion_matrix": cm.tolist(),
    "target_class": TARGET_CLASS
}
with open("evaluation_results_binary.json", "w") as f:
    json.dump(results, f, indent=2)

print("Avaliação concluída. Arquivos gerados: matriz_confusao_db_binary.png, grafico_historico_binary.png, evaluation_results_binary.json")
