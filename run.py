import cv2
from PIL import Image
import sys

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"
maskPath = "mask.png"

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.15) 

background = Image.open(imagePath)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('face detected', image)
    cv2.waitKey(0)

    mask = Image.open(maskPath)
    mask = mask.resize((w, h), Image.ANTIALIAS)
    
    offset = (x,y)
    background.paste(mask, offset, mask=mask)

background.save('out.png')
