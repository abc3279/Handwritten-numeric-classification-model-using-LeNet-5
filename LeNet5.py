# import
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

# preparing data
train_ds = tf.keras.utils.image_dataset_from_directory(
    './mask_trainset', image_size=(227, 227), seed=1337, batch_size=128, label_mode='binary', shuffle=True, validation_split=0.1, subset='training')

val_ds = tf.keras.utils.image_dataset_from_directory(
    './mask_trainset', image_size=(227, 227), seed=1337, batch_size=128, label_mode='binary', shuffle=True, validation_split=0.1, subset='validation')

test_ds = tf.keras.utils.image_dataset_from_directory('./mask_testset', image_size=(227, 227), batch_size=20, label_mode='binary')

# create model
X = tf.keras.layers.Input(shape = [227, 227, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='tanh')(X) #LeNet5 에서는 32x32 이미지를 사용했지만, mnist는 28x28이라 padding을 사용해서 줄이지 않아야 같아진다.
H = tf.keras.layers.AvgPool2D()(H)

H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='tanh')(H) # 필터의 최종 갯수는 같으나 파라미터가 맞지 않음.
H = tf.keras.layers.AvgPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='tanh')(H) # 다시 파라미터가 같아짐.
H = tf.keras.layers.Dense(84, activation='tanh')(H)

Y = tf.keras.layers.Dense(1, activation='sigmoid')(H)

model = tf.keras.models.Model(X, Y)

SGD = tf.keras.optimizers.SGD(learning_rate=0.0005) # 원문에서는 epoch 당 lr이 감소
# Adam = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', metrics='acc', optimizer=SGD)

print("Fit: ")
model.fit(train_ds, validation_data=val_ds, epochs=300, batch_size=128) # 논문에는 epoch 20. 하지만 SGD의 hyper parameter 값을 모르겠어서 epoch 조정.

print("Evalute: ")
score = model.evaluate(test_ds, batch_size=128)
print("정답률 = ", score[1], 'loss = ', score[0])

print("실제 사진 분류: ")
predict = model.predict(test_ds.take(1))
print(pd.DataFrame(predict).round(3))