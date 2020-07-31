import cv2
import imageio

# imageio can only transform png to gif
# so transform tif to png first

path = './data/Fluo-N2DL-HeLa/01/gen/'
name_list = [str(i) for i in range(1, 21)]  # get the frame number
img_list = [path + i + '.tif' for i in name_list]
orginal = []
for img in img_list:
    orginal.append(cv2.imread(img, -1))

png_list = [path + i + '.png' for i in name_list]

for index, img in enumerate(png_list):
    cv2.imwrite(img, orginal[index])

frames = []
for img in png_list:
    frames.append(imageio.imread(img))

imageio.mimsave('tracking.gif', frames, 'GIF', duration=0.5)  # duration is the interval for each frame (s)
