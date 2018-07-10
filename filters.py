#!/usr/bin/python3

import cv2

# img = cv2.imread('../others/lenna.jpg')

imgHeight, imgWidth = img.shape[0]//2, img.shape[1]//2
img = cv2.resize(img, (imgWidth, imgHeight))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_inv_blur = cv2.GaussianBlur(255-gray, (21, 21), 0, 0)
gray_blend = cv2.divide(gray, 255-gray_inv_blur, scale=256) # dodge

cv2.imshow('img', gray_blend)
cv2.waitKey()
cv2.destroyAllWindows()
