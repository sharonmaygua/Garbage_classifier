import cv2
import numpy as np
from Clasificador import Classifier
from Video import Video

def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640,240)
cv2.createTrackbar("Threshold1", "Parameters", 115,255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 48,255, empty)
cv2.createTrackbar("area", "Parameters", 5000,30000, empty)

def main():
    video = Video(0)
    classifier = Classifier('keras_model.h5', 'labels.txt')
    while True:
        exito, img = video.leerFrame()

        imgContour = img.copy()

        predection = classifier.getPrediction(imgContour) #cambio
        print(predection)

        imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        # hasta aca ya puedo ver los bordes pero tienen ruido asi que vamos a utilizar la duncion dilation

        # dilation
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)  # esto me pone los bordes mas grusesos
        contornos = classifier.getContours(imgDil, imgContour)  # tiene que estar aca antes del stack

        if not exito:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #predection = classifier.getPrediction(imgContour) #cambio
        #print(predection)
        video.mostrarFrame("Clasificador de basura", imgContour)
        if cv2.waitKey(1) == ord('q'):
            break
    video.terminarVideo()
    classifier.closeSerial()

if __name__ == "__main__":
    main()