import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


img = scipy.ndimage.imread("./humerus_detector/samples/1.jpg")
img = rgb2gray(img)

print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()



