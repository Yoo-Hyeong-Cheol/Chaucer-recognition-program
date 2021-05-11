import os
import cv2
from glob import glob
from PIL import Image
import numpy as np
from keras import Input
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split

image_dir_path = os.path.dirname(os.path.abspath("__file__")) + '/image'
categories = os.listdir('./image')
select_cat = []
nb_classes = 0
mnist = tf.keras.datasets.mnist
image_w = 68
image_h = 68

x = []
y = []

count = 0;
for idx, c in enumerate(categories):
    dir = image_dir_path + '/' + c
    f = glob(dir+'/*.jpg')
    
    if len(f) >= 200:
        nb_classes += 1
        select_cat.append(dir)
        
for index, cat in enumerate(select_cat):
    files = glob(select_cat[index]+'/*.jpg')
    for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("L")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            
            x.append(data)
            y.append(index)

x = np.array(x)
y = np.array(y)

x_train = x
y_train = y

x_train.shape, y_train.shape

x_train = x_train.reshape(-1, image_w, image_h, 1)
x_train = x_train / 127.5 - 1
print(x_train.shape)
x_train.min(), x_train.max()

encoder_input = Input(shape=(image_w, image_h, 1))

# 28 X 28
x = Conv2D(32, 3, padding='same')(encoder_input) 
x = BatchNormalization()(x)
x = LeakyReLU()(x) 

# 28 X 28 -> 14 X 14
x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x) 
x = LeakyReLU()(x) 

# 14 X 14 -> 7 X 7
#x = Conv2D(64, 3, strides=2, padding='same')(x)
#x = BatchNormalization()(x)
#x = LeakyReLU()(x)

# 17 X 7
x = Conv2D(64, 3, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

# 2D 좌표로 표기하기 위하여 2를 출력값으로 지정합니다.
encoder_output = Dense(2)(x)

encoder = Model(encoder_input, encoder_output)
encoder.summary()

# Input으로는 2D 좌표가 들어갑니다.
decoder_input = Input(shape=(2, ))

# 2D 좌표를 7*7*64 개의 neuron 출력 값을 가지도록 변경합니다.
x = Dense(34*34*64)(decoder_input)
x = Reshape((34, 34, 64))(x)

# 7 X 7 -> 7 X 7
x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 7 X 7 -> 14 X 14
#x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
#x = BatchNormalization()(x)
#x = LeakyReLU()(x)

# 14 X 14 -> 28 X 28
x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 28 X 28 -> 28 X 28
x = Conv2DTranspose(32, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 최종 output
decoder_output = Conv2DTranspose(1, 3, strides=1, padding='same', activation='tanh')(x)

decoder = Model(decoder_input, decoder_output)
decoder.summary()

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
encoder_in = Input(shape=(image_w, image_h, 1))
x = encoder(encoder_in)
decoder_out = decoder(x)

auto_encoder = Model(encoder_in, decoder_out)
auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=tf.keras.losses.MeanSquaredError())

auto_encoder.fit(x_train, x_train, 
                 batch_size=BATCH_SIZE, 
                 epochs=50
                )

decoded_images = auto_encoder.predict(x_train)

decoded_images = decoded_images.reshape(3546, image_w, image_h)
decoded_images.shape
for i, idx in enumerate(y_train):
    im = (decoded_images[i]*255).astype(np.uint8)
    image = Image.fromarray(im, "L")
    name = select_cat[idx][44:]+str(i)
    print(name)
    image.save("C:\\Users\\HYEONGCHEOL\\Desktop\\TestFile/Auto_encoder_image/"+select_cat[idx][44:]+"/"+name+".jpg", "JPEG")
    i += i
    
fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    axes[i//5, i%5].imshow(x_train[i].reshape(image_w, image_h), cmap='gray')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Original Images')
plt.show()

fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    axes[i//5, i%5].imshow(decoded_images[i].reshape(image_w, image_h), cmap='gray')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Auto Encoder Images')
plt.show()
