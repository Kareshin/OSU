import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50), dtype=np.uint8)
white = np.ones((50, 50), dtype=np.uint8)

top = np.hstack((black, white))
bottom = np.hstack((white, black))
final = np.vstack((top, bottom))

plt.imshow(final, cmap="gray")
plt.show()