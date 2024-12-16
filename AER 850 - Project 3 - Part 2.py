!pip install ultralytics
import os
import numpy as np
import cv2
from ultralytics import YOLO

train_dir = '/content/drive/MyDrive/Project_3_Data/train/images'
eval_dir = '/content/drive/MyDrive/Project_3_Data/evaluation'

model = YOLO('/content/drive/MyDrive/Project_3_Data/yolov8s.pt')

epochs = 150
batch = 100
img_size = 120

model.train(data='/content/drive/MyDrive/Project_3_Data/data.yaml', epochs=epochs, batch=batch, imgsz=img_size)


img1 = cv2.imread(os.path.join(eval_dir, 'ardmega.jpg'))
img2 = cv2.imread(os.path.join(eval_dir, 'arduno.jpg'))
img3 = cv2.imread(os.path.join(eval_dir, 'rasppi.jpg'))


outputs1 = model.predict(img1)
outputs2 = model.predict(img2)
outputs3 = model.predict(img3)


print('Image 1:')
print(outputs1[0])
print('Image 2:')
print(outputs2[0])
print('Image 3:')
print(outputs3[0])


img1_detected_components = []
img2_detected_components = []
img3_detected_components = []


for detection in outputs1[0].boxes.data.tolist(): 
    class_id = int(detection[5]) 
    confidence = detection[4] 
    if confidence > 0.5:
        img1_detected_components.append(class_id)

for detection in outputs2[0].boxes.data.tolist():
    class_id = int(detection[5])
    confidence = detection[4]
    if confidence > 0.5:
        img2_detected_components.append(class_id)

for detection in outputs3[0].boxes.data.tolist():
    class_id = int(detection[5])
    confidence = detection[4]
    if confidence > 0.5:
        img3_detected_components.append(class_id)

print('Image 1: Detected components:', img1_detected_components)
print('Image 2: Detected components:', img2_detected_components)
print('Image 3: Detected components:', img3_detected_components)


accuracy = 0
for image in img1_detected_components + img2_detected_components + img3_detected_components:
    accuracy += 1
accuracy /= 3
print('Accuracy:', accuracy)


outputs1 = model.predict(img1)
outputs2 = model.predict(img2)
outputs3 = model.predict(img3)

def draw_bounding_boxes(image, outputs):
    for detection in outputs[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = map(int, detection[:6])
        if confidence > 0.5:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Class {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

img1_with_boxes = draw_bounding_boxes(img1.copy(), outputs1)
img2_with_boxes = draw_bounding_boxes(img2.copy(), outputs2)
img3_with_boxes = draw_bounding_boxes(img3.copy(), outputs3)


img1_rgb = cv2.cvtColor(img1_with_boxes, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2_with_boxes, cv2.COLOR_BGR2RGB)
img3_rgb = cv2.cvtColor(img3_with_boxes, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1_rgb)
plt.title("Arduino Mega w/ Detected Components")

plt.subplot(1, 3, 2)
plt.imshow(img2_rgb)
plt.title("Arduino Uno w/ Detected Components")

plt.subplot(1, 3, 3)
plt.imshow(img3_rgb)
plt.title("Raspberry Pi w/ Detected Components")

plt.show()
