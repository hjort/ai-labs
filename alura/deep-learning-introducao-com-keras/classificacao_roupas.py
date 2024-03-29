# -*- coding: utf-8 -*-
"""classificacao_roupas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17rYMEuSD4MdGw68pXwnBlZJ1lz8uvmaU
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

"""- https://github.com/zalandoresearch/fashion-mnist"""

dataset = keras.datasets.fashion_mnist

((imagens_treino, identificacoes_treino), (imagens_teste, identificacoes_teste)) = dataset.load_data()

imagens_treino[:2]

len(imagens_treino)

imagens_treino.shape

imagens_teste.shape

len(identificacoes_treino)

plt.imshow(imagens_treino[0])
plt.title(identificacoes_treino[0])

plt.imshow(imagens_treino[1])
plt.title(identificacoes_treino[1])

identificacoes_treino.min()

identificacoes_treino.max()

total_classificacoes = identificacoes_treino.max() - identificacoes_treino.min() + 1
total_classificacoes

nomes_classificacoes = ['Camiseta', 'Calça', 'Pullover', 'Vestido', 'Casaco',
                        'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']
#nomes_classificacoes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
nomes_classificacoes

for imagem in range(15):
  plt.subplot(3, 5, imagem + 1)
  plt.imshow(imagens_treino[imagem])
  plt.title(nomes_classificacoes[identificacoes_treino[imagem]])

plt.imshow(imagens_treino[0])
plt.colorbar()

modelo = keras.Sequential([

    # entrada (camada 0)
    keras.layers.Flatten(input_shape=(28, 28)), # achatar imagens de 28 x 28 pixels

    # processamento (camada 1)
    keras.layers.Dense(256, activation=tf.nn.relu), # função ReLU (não-linear)

    # saída (camada 2)
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10 categorias
])

modelo

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

modelo.fit(imagens_treino, identificacoes_treino) # loss: 3.9652

# normalizar imagens para diminuir a perda
imagens_treino = imagens_treino / float(255)

modelo.fit(imagens_treino, identificacoes_treino) # loss: 0.8597

modelo = keras.Sequential([

    # entrada (camada 0)
    keras.layers.Flatten(input_shape=(28, 28)), # achatar imagens de 28 x 28 pixels

    # processamento (camadas ocultas)
    keras.layers.Dense(256, activation=tf.nn.relu), # função ReLU (não-linear)
    keras.layers.Dense(128, activation=tf.nn.relu), # função ReLU (não-linear)

    # saída (camada N)
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10 categorias
])
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

modelo.fit(imagens_treino, identificacoes_treino) # loss: 0.4755

modelo = keras.Sequential([

    # entrada (camada 0)
    keras.layers.Flatten(input_shape=(28, 28)), # achatar imagens de 28 x 28 pixels

    # processamento (camadas ocultas)
    keras.layers.Dense(256, activation=tf.nn.relu), # função ReLU (não-linear)
    keras.layers.Dense(128, activation=tf.nn.relu), # função ReLU (não-linear)
    keras.layers.Dense(64, activation=tf.nn.relu), # função ReLU (não-linear)

    # saída (camada N)
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10 categorias
])
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

modelo.fit(imagens_treino, identificacoes_treino) # loss: 0.4839 => não está mais ajudando incluir camadas!!!

# aumentar quantidade de épocas
modelo.fit(imagens_treino, identificacoes_treino, epochs=5) # loss: 0.2755

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# executar 5 épocas e separar 20% para validação
historico = modelo.fit(imagens_treino, identificacoes_treino, epochs=5, validation_split=0.2) # loss: 0.2220

id_teste = 1
testes = modelo.predict(imagens_teste)
print('resultado teste:', np.argmax(testes[id_teste]))
print('número da imagem de teste:', identificacoes_teste[id_teste])

# avaliar como o modelo está indo com os dados de teste, fornecendo os dados de acurácia e perda
perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print('Perda do teste:', perda_teste)
print('Acurácia do teste:', acuracia_teste * 100)

historico.history

modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # entrada (camada 0)
    keras.layers.Dense(256, activation=tf.nn.relu), # (camada 1)
    keras.layers.Dropout(0.2), # para normalizar o modelo (camada 2)
    keras.layers.Dense(10, activation=tf.nn.softmax) # saída (camada 3)
])

modelo.compile(optimizer='adam', 
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# gravar o histórico de treino do modelo
historico = modelo.fit(imagens_treino, identificacoes_treino, epochs=5, validation_split=0.2)

# visualizar a acurácia de treino e validação
plt.plot(historico.history['acc'])
plt.plot(historico.history['val_acc'])
plt.title('Acurácia por épocas')
plt.xlabel('épocas')
plt.ylabel('acurácia')
plt.legend(['treino', 'validação'])

# visualizar a perda de treino e validação
plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Perda por épocas')
plt.xlabel('épocas')
plt.ylabel('perda')
plt.legend(['treino', 'validação'])

perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print('Perda do teste:', perda_teste)
print('Acurácia do teste:', acuracia_teste * 100)

from tensorflow.keras.models import load_model

modelo.save('modelo.h5') # gravar no formato HDF5
modelo_salvo = load_model('modelo.h5')

testes = modelo.predict(imagens_teste)
print('resultado teste:', np.argmax(testes[1]))
print('número da imagem de teste:', identificacoes_teste[1])

testes_modelo_salvo = modelo_salvo.predict(imagens_teste)
print('resultado teste modelo salvo:', np.argmax(testes_modelo_salvo[1]))
print('número da imagem de teste:', identificacoes_teste[1])

