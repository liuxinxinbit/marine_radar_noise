from virtual_data import get_train_data
from RTNet import RTNet
import numpy as np
from tensorflow.keras.models import load_model, save_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random

rtnet = RTNet()
# rtnet.train(epochs=10, steps_per_epoch=1000, batch_size=12)
# rtnet.save()


rtnet.load()
for flag in range(500):
    images,truths = get_train_data(5)
    image = images[0,:,:,:]
    label = truths[0,:,:]
    plt.subplot(1, 3, 1)
    plt.imshow(label)
    prediction = rtnet.predict(image)

    plt.subplot(1, 3, 2)
    plt.imshow(prediction[0,:,:,0])
    plt.subplot(1, 3, 3)
    plt.imshow(image[:,:,0])
    plt.pause(0.01)
    # plt.clf()
    plt.show()