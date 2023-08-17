import tensorflow.keras
import numpy as np
import cv2
import serial
import time
from pickle import TRUE



class Classifier:
    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        self.arduino = serial.Serial('COM4', 9600, timeout=1)

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        # cargar el modelo
        self.model = tensorflow.keras.models.load_model(self.model_path)
        # creacion de matriz de imagenes para alimenar el modelo de keras
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = []
            for line in label_file:
                stripped_line = line.strip()
                self.list_labels.append(stripped_line)
            label_file.close()
        else:
            print("No se encontraron etiquetas")


    def getPrediction(self, img, draw= True, pos=(50, 50), scale=1, color = (0,0,255)): #cambio en parametros
        # redimensionar imagen a 224x224
        imgS = cv2.resize(img, (224, 224))
        # convertir images a numpy array
        image_array = np.asarray(imgS)
        # normalizar imagen
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # cargar imagen al array
        self.data[0] = normalized_image_array


        # correr interfaz
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)


        #parte de bordes


        if draw and self.labels_path:


            if (str(self.list_labels[indexVal])=='0 plastic'):
                self.arduino.write(b'p') # plastico
            elif (str(self.list_labels[indexVal])=='1 paper'):
                self.arduino.write(b'h')  # papel
            elif (str(self.list_labels[indexVal])=='2 biological'):
                self.arduino.write(b'b')  #biodegradable
            elif (str(self.list_labels[indexVal])=='3 metal'):
                self.arduino.write(b'm')  # metal


            cv2.putText(img, str(self.list_labels[indexVal]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)
            return list(prediction[0]), indexVal

    def getContours(self, img, imgContour):  # imput y oitput image
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # para reducir el ruido
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areaMin = cv2.getTrackbarPos("area", "Parameters")
            if area > areaMin:
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(imgContour, "Vertices: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            (0, 255, 0), 2)
                cv2.putText(imgContour, "Area: " + str(area), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 2)


    def closeSerial(self):
        self.arduino.close()




