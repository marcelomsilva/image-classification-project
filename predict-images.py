import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Caminhos
model_path = "modelo_flores.keras"
dataset_path = "dataset_flores"
img_size = (224, 224)  # Mesmo tamanho usado no treinamento

# ============================
# ETAPA 1 – CARREGAR MODELO
# ============================
model = tf.keras.models.load_model(model_path)

# ============================
# ETAPA 2 – MAPEAR CLASSES
# ============================
# Pega os nomes das classes com base na estrutura da pasta
# class_names = sorted(os.listdir(dataset_path))
class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# ============================
# ETAPA 3 – PREPARAR IMAGEM
# ============================
# Caminho da imagem a ser testada
image_path = "/home/marcelo/Documents/estudos/projetos/flower_photos/rosa.jpg"

# Carrega e redimensiona a imagem
img = image.load_img(image_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão de batch

# ============================
# ETAPA 4 – FAZER PREDIÇÃO
# ============================
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class] * 100

# ============================
# ETAPA 5 – EXIBIR RESULTADO
# ============================
print(f"Classe prevista: {class_names[predicted_class]}")
print(f"Confiança: {confidence:.2f}%")
