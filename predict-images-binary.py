import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ============================
# CONFIGURAÇÕES
# ============================
# Caminho do modelo salvo (ajuste se necessário)
model_path = "modelo_rosa.keras"  # ou "modelo_flores.keras" se for o caso

# Classe alvo usada durante o treino (string tal como está no banco)
TARGET_CLASS = "roses"

# Tamanho usado no treinamento
IMG_SIZE = (224, 224)

# Threshold para decidir positivo (1) vs negativo (0)
THRESHOLD = 0.5

# Caminho da imagem a ser testada (ajuste)
image_path = "/home/marcelo/Documents/estudos/projetos/flower_photos/sunflower-1.jpg"

# ============================
# ETAPA 1 – CARREGAR MODELO
# ============================
model = tf.keras.models.load_model(model_path)
model.summary()  # opcional: remove se não quiser ver o resumo

# ============================
# ETAPA 2 – MAPEAR RÓTULOS BINÁRIOS
# ============================
NEG_LABEL = f"not_{TARGET_CLASS}"
POS_LABEL = TARGET_CLASS
BINARY_LABELS = [NEG_LABEL, POS_LABEL]

# ============================
# ETAPA 3 – PREPARAR IMAGEM
# ============================
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

# Carrega e redimensiona a imagem
img = image.load_img(image_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)  # shape (H, W, 3), dtype float32
# expand para batch
img_batch = np.expand_dims(img_array, axis=0)

# Aplica preprocess_input compatível com EfficientNetV2
img_batch_proc = preprocess_input(img_batch)

# ============================
# ETAPA 4 – FAZER PREDIÇÃO (robusto para várias shapes de saída)
# ============================
preds = model.predict(img_batch_proc, verbose=0)  # pode ser shape (1,1), (1,), (1,2), etc.
preds = np.asarray(preds)

# Normaliza para vetor 1D de probabilidades do POS_LABEL
if preds.ndim == 2 and preds.shape[1] == 1:
    prob_pos = preds.reshape(-1)[0]
elif preds.ndim == 1:
    prob_pos = preds[0]
elif preds.ndim == 2 and preds.shape[1] == 2:
    # modelo com softmax de 2 saídas: interpretamos coluna 1 como prob do target
    prob_pos = preds[0, 1]
else:
    # Caso inesperado: tenta reduzir para um escalar
    try:
        prob_pos = float(np.mean(preds))
        print("Warning: formato de saída inesperado. Usando média das saídas como probabilidade.")
    except Exception:
        raise ValueError(f"Formato de output do modelo não reconhecido: shape={preds.shape}")

# Converte para binário com threshold
pred_bin = 1 if prob_pos >= THRESHOLD else 0
pred_label = POS_LABEL if pred_bin == 1 else NEG_LABEL
confidence = prob_pos * 100.0 if pred_bin == 1 else (1.0 - prob_pos) * 100.0

# ============================
# ETAPA 5 – EXIBIR RESULTADO
# ============================
print("=== Resultado da predição (modelo binário) ===")
print(f"Imagem: {image_path}")
print(f"Classe prevista: {pred_label} (binário: {pred_bin})")
print(f"Probabilidade de {POS_LABEL}: {prob_pos:.4f}")
print(f"Confiança (na classe prevista): {confidence:.2f}%")
print(f"Threshold usado: {THRESHOLD}")
