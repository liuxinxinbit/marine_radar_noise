from virtual_data import get_train_data
from RTNet import RTNet
import numpy as np
from tensorflow.keras.models import load_model, save_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random

rtnet = RTNet()
rtnet.load()
rtnet.train(epochs=5, steps_per_epoch=500, batch_size=24)
rtnet.save()


rtnet.load()
for flag in range(500):
    images,truths = get_train_data(5)
    image = images[0,:,:,:]
    label = truths[0,:,:]
    plt.subplot(1, 3, 1)
    plt.title("label")
    plt.imshow(label)
    prediction = rtnet.predict(image)

    plt.subplot(1, 3, 2)
    plt.title("prediction")
    plt.imshow(prediction[0,:,:,0]>0.1)
    plt.subplot(1, 3, 3)
    plt.title("image")
    plt.imshow(image[:,:,0])#+image[:,:,1]+image[:,:,2]+image[:,:,3]+image[:,:,4])
    plt.pause(0.5)
    # plt.clf()
    plt.show()