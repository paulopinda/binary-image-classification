import sys
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

path_image = sys.argv[1]


# Dimensões das imagens.
img_width, img_height = 128, 128

# Carrega o modelo gerado no treinamento.
model = load_model('my_model.h5')
model.compile(
    loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Predição para classificação da imagem.
img = image.load_img(path_image, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)

print(classes)

# if classes[0][0] == 0:
#    print('Carro.')

# if classes[0][0] == 1:
#    print('Caminhão.')


