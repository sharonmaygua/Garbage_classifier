import numpy as np
import cv2


class Video:
    def __init__(self, fuenteVideo):
        self.__video = cv2.VideoCapture(fuenteVideo)
        self.__frame = None
        if not self.__video.isOpened():
            print("Cannot open camera")
            exit()

    def leerFrame(self):
        ret, frame = self.__video.read()
        self.__frame = frame
        return ret, self.__frame


    def mostrarFrame(self, nombreVentana, imgContour): #cambio
        cv2.imshow(nombreVentana, imgContour) #contorno cambio en frame

    def terminarVideo(self):
        self.__video.release()
        cv2.destroyAllWindows()

