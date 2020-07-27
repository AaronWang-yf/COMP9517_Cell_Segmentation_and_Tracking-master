import cv2
import matplotlib.pyplot as plt
import numpy as np




true_path = './man_mask/Fluo-N2DL-HeLa/01_ST/man_track000.tif'
detect_path ='./data/Fluo-N2DL-HeLa/01/gen/mask_1.tif'
img_true= cv2.imread(true_path,-1)
img_detect = cv2.imread(detect_path,-1)

img_true[img_true > 0 ] = 255
img_detect = img_detect.astype(np.uint16)

plt.imshow(img_true, cmap=plt.cm.nipy_spectral)
plt.show()

plt.imshow(img_detect, cmap=plt.cm.nipy_spectral)
plt.show()


union = img_true + img_detect
union[union > 0 ] = 255
intersect = img_true + img_detect - union
union = union.astype(np.uint8)
intersect = intersect.astype(np.uint8)
plt.imshow(union, cmap=plt.cm.nipy_spectral)
plt.show()

plt.imshow(intersect, cmap=plt.cm.nipy_spectral)
plt.show()

contour1,_ = cv2.findContours(union, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour2,_ = cv2.findContours(intersect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


print(len(contour2)/len(contour1))

