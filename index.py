import glob
import cv2
import os
import descriptor
from abc import abstractmethod, ABCMeta


class AbstractIndexBuilder(metaclass=ABCMeta):
    @abstractmethod
    def build(index_file_path, image_set_path):
        pass


class ColorIndexBuilder(AbstractIndexBuilder):
    @staticmethod
    def build(index_file_path, image_set_path):
        colorDesriptor = descriptor.ColorDescriptor()
        output = open(index_file_path, "w")
        images = glob.glob(image_set_path)
        print("Find images: " + str(len(images)))
        for imagePath in images:
            imageName = os.path.basename(imagePath)
            image = cv2.imread(imagePath)
            features = colorDesriptor.describe(image)
            # write features to file
            features = [str(feature).replace("\n", "") for feature in features]
            output.write("%s,%s\n" % (imageName, ",".join(features)))
            print("build color index --> " + imageName)
        # close index file
        output.close()


class StructureIndexBuilder(AbstractIndexBuilder):
    @staticmethod
    def build(index_file_path, image_set_path):
        structureDescriptor = descriptor.StructureDescriptor()
        output = open(index_file_path, "w")
        images = glob.glob(image_set_path)
        print("Find images: " + str(len(images)))
        for imagePath in images:
            imageName = os.path.basename(imagePath)
            image = cv2.imread(imagePath)
            structures = structureDescriptor.describe(image)
            # write structures to file
            structures = [
                str(structure).replace("\n", "") for structure in structures
            ]
            output.write("%s,%s\n" % (imageName, ",".join(structures)))
            print("build structure index --> " + imageName)
        # close index file
        output.close()
