import cv2
import matplotlib.pyplot as plt
import numpy as np

path = './data/Fluo-N2DL-HeLa/Sequence 1/t001.tif'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

plt.figure('test1')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('test1')
plt.show()

kernel = 11
img = cv2.GaussianBlur(img, (kernel, kernel), 0)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
th = np.where(hist == np.max(hist))
ret, thresh = cv2.threshold(img, th[0] + 1, 255, cv2.THRESH_BINARY)
thresh = cv2.erode(thresh, np.ones((9, 9)))
thresh = cv2.dilate(thresh, np.ones((9, 9)))
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.show()

thresh, contours, hirearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
original = cv2.imread(path,1)

plt.imshow(original)
plt.axis('off')
plt.show()

draw = cv2.drawContours(original, contours, -1, (0, 255, 0), 2)

plt.imshow(original)
plt.axis('off')
plt.show()
cv2.imwrite('test.tif',original)