import cv2

im_gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_OCEAN)
cv2.imwrite('test2.jpg',im_color)