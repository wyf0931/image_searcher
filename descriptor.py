import cv2
import numpy
from abc import abstractmethod, ABCMeta


class AbstractDescriptor(metaclass=ABCMeta):
    @abstractmethod
    def describe(self, image):
        pass


class StructureDescriptor(AbstractDescriptor):
    __slot__ = ["dimension"]

    def __init__(self, dimension=(16, 16)):
        self.dimension = dimension

    def describe(self, image):
        image = cv2.resize(
            image, self.dimension, interpolation=cv2.INTER_CUBIC)
        return image


class ColorDescriptor(AbstractDescriptor):
    __slot__ = ["bins"]

    def __init__(self, bins=(8, 12, 3)):
        self.bins = bins

    def getHistogram(self, image, isCenter):
        # get histogram
        imageHistogram = cv2.calcHist([image], [0, 1, 2], None, self.bins,
                                      [0, 180, 0, 256, 0, 256])
        # normalize
        imageHistogram = cv2.normalize(imageHistogram,
                                       imageHistogram).flatten()
        if isCenter:
            weight = 5.0
            for index in range(len(imageHistogram)):
                imageHistogram[index] *= weight
        return imageHistogram

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        # get dimension and center
        height, width = image.shape[0], image.shape[1]
        centerX, centerY = int(width * 0.5), int(height * 0.5)
        # initialize mask dimension
        segments = [(0, centerX, 0, centerY), (0, centerX, centerY, height),
                    (centerX, width, 0, centerY), (centerX, width, centerY,
                                                   height)]
        # initialize center part
        axesX, axesY = int(width * 0.75 / 2), int(height * 0.75 / 2)
        ellipseMask = numpy.zeros([height, width], dtype="uint8")

        center = (centerX, centerY)
        axes = (axesX, axesY)
        color = (255, 255, 255)
        cv2.ellipse(ellipseMask, center, axes, 0, 0, 360, color, -1)
        # initialize corner part
        for startX, endX, startY, endY in segments:
            cornerMask = numpy.zeros([height, width], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipseMask)
            # get histogram of corner part
            imageHistogram = self.getHistogram(image, False)
            features.append(imageHistogram)
        # get histogram of center part
        imageHistogram = self.getHistogram(image, True)
        features.append(imageHistogram)
        return features
