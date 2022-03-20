import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')

for x in range(1,6):
  img = cv.imread(f'{x}.png')[:,:,0]
  img = np.invert(np.array([img]))
  prediction = model.predict(img)
  predict = np.argmax(prediction) 
  if predict == 0:
    clothe = "T-shirt/Top"
  if predict == 1:
    clothe = "Trouser"
  if predict == 2:
    clothe = "Pullover"
  if predict == 3:
    clothe = "Dress"
  if predict == 4:
    clothe = "Coat"
  if predict == 5:
    clothe = "Sandal"
  if predict == 6:
    clothe = "Shirt"
  if predict == 7:
    clothe = "Sneaker"
  if predict == 8:
    clothe = "Bag"
  if predict == 9:
    clothe = "Ankle boot"
  print('The picture is probably a: ' + clothe)
  plt.imshow(img[0], cmap=plt.cm.binary)
  plt.show()
  plt.bar(range(10), prediction[0])
  plt.show()
  print("prediction: class", np.argmax(prediction[0]))
  
