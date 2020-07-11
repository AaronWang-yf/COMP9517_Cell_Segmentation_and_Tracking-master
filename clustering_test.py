# %%

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

img = cv2.imread('./data/DIC-C2DH-HeLa/Sequence 1/t002.tif', cv2.IMREAD_UNCHANGED)

# plt.figure('test1')
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.title('test1')
# plt.show()

hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.show()

# %%


sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # x方向的梯度
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # y方向的梯度

sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

sobelCombined = cv2.bitwise_or(sobelX, sobelY)  #
# plt.figure('sob')
# plt.imshow(sobelCombined, cmap='gray')
# plt.axis('off')
# plt.title('sob')
# plt.show()

hist, bins = np.histogram(sobelCombined.flatten(), 256, [0, 256])
# plt.hist(sobelCombined.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.show()

# %%

ret, thresh = cv2.threshold(sobelCombined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8))
D = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3)))
# plt.imshow(D, cmap='gray')
# plt.show()

####  connect
connectivity = 8  # or whatever you prefer
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(D, connectivity, cv2.CV_32S)

sizes = stats[1:, -1]
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 0

# your answer image
img2 = np.zeros((output.shape))
# for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

## watershed
# img2 = cv2.dilate(img2,np.ones((3,3)),iterations=3)
# img2 = cv2.erode(img2,np.ones((3,3)),iterations=3)
plt.imshow(img2, cmap='gray')
plt.show()
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(img2)
# plt.imshow(distance, cmap='gray')
# plt.show()
new_t = np.zeros(distance.shape)
cv2.normalize(distance, new_t, 0, 255, cv2.NORM_MINMAX)
new_t = np.array(new_t,dtype=np.uint8)
ret, new_t = cv2.threshold(new_t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

new_t = cv2.dilate(new_t,np.ones((3, 3), np.uint8))
new_t = cv2.morphologyEx(new_t, cv2.MORPH_OPEN, np.ones((4, 4)))
plt.imshow(new_t, cmap='gray')
plt.show()


new_t, contours, hierarchy = cv2.findContours(new_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def draw_approx_hull_polygon(img, cnts):
    # img = np.copy(img)
    img = np.zeros(img.shape, dtype=np.uint8)

    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue

    epsilion = img.shape[0]/32
    approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
    cv2.polylines(img, approxes, True, (0, 255, 0), 2)  # green

    hulls = [cv2.convexHull(cnt) for cnt in cnts]
    cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red

    return img
output = draw_approx_hull_polygon(new_t, contours)

plt.imshow(output)
plt.show()