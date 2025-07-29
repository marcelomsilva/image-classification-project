# Importa as bibliotecas necessárias
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Caminhos dos arquivos e configurações
history_path = "historico_treinamento.json"      # Arquivo com histórico de acurácia e perda
model_path = "modelo_flores.keras"               # Caminho do modelo salvo
dataset_path = "/home/marcelo/Documents/estudos/projetos/flower_photos"                  # Pasta com as imagens organizadas por classe
img_size = (224, 224)                            # Tamanho padrão de entrada para EfficientNetB7
batch_size = 32                                  # Número de imagens por batch

# ========================
# ETAPA 1 – HISTÓRICO DE TREINAMENTO
# ========================
# Carrega os dados de treinamento salvos durante o treino do modelo
with open(history_path, 'r') as f:
    history = json.load(f)

# Cria dois gráficos lado a lado: acurácia e perda
plt.figure(figsize=(12, 5))

# Gráfico da acurácia
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Treino')
plt.plot(history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Gráfico da perda (loss)
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Treino')
plt.plot(history['val_loss'], label='Validação')
plt.title('Perda (Loss) por Época')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

# Exibe os dois gráficos
plt.tight_layout()
plt.savefig("grafico_historico.png")
plt.show()

# ========================
# ETAPA 2 – AVALIAÇÃO EM DETALHE (PÓS-TREINAMENTO)
# ========================

# Carrega o modelo treinado
model = tf.keras.models.load_model(model_path)

# Cria o gerador de dados apenas para validação
# O mesmo que foi usado durante o treino, com a mesma proporção
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2  # Deve ser igual ao que foi usado no treino
)

# Gera os dados da validação, SEM embaralhar, para bater as classes depois
val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Faz predições com o modelo para todas as imagens da validação
predictions = model.predict(val_gen)

# Converte as saídas do modelo (vetores de probabilidades) para classes (índices)
y_pred = np.argmax(predictions, axis=1)

# Pega os rótulos verdadeiros que o gerador já conhece
y_true = val_gen.classes

# Obtém os nomes das classes (ex: ['margarida', 'rosa', 'tulipa', ...])
# class_labels = list(val_gen.class_indices.keys())
# print(class_labels)

class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips', 'classe_desconhecida']


# ========================
# MATRIZ DE CONFUSÃO
# ========================
# Compara as predições com os valores reais e gera a matriz
cm = confusion_matrix(y_true, y_pred)
print(cm.shape)

# Plota a matriz de confusão com nomes das classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.savefig("matriz-confusao.png")
plt.show()

# ========================
# RELATÓRIO DE CLASSIFICAÇÃO
# ========================
# Mostra métricas como precisão, recall e F1-score por classe
report = classification_report(y_true, y_pred, target_names=class_labels)
print("===== RELATÓRIO DE CLASSIFICAÇÃO =====")
print(report)
