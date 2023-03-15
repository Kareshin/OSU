import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")

#a)
brightened_img = np.clip(img.astype(np.int32) + 50, 0, 255).astype(np.uint8)
plt.imshow(img)
plt.show()
plt.imshow(brightened_img)
plt.show()

#b)
height, width, unused = img.shape
cropped_img = img[:, width // 4:width // 2, :]
plt.imshow(cropped_img)
plt.show()

#c)
rotated_img = np.rot90(img, k=-1)
plt.imshow(rotated_img)
plt.show()

#d)
flipped_image = np.fliplr(img)
plt.imshow(flipped_image)
plt.show()