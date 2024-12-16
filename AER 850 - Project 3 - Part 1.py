import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/Users/aadhi/Documents/GitHub/aadhith/motherboard_image.JPEG"
original_image = cv2.imread(image_path)

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

edges = cv2.Canny(blurred_image, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

mask = np.zeros_like(gray_image)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

extracted_image = cv2.bitwise_and(original_image, original_image, mask=mask)
colored_extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(edges, cmap="gray")
plt.title("Edge Detection Output")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(mask, cmap="gray")
plt.title("Mask Image")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(colored_extracted_image)
plt.title("Final Extracted Image")
plt.axis("off")
plt.show()
