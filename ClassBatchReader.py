# -*- coding: utf-8 -*-
import math
import os
import random
import numpy
from sklearn.utils import shuffle
from matplotlib import image
import json
import tensorflow as tf
import sys
import cv2

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

image_sizeRows = 96#192#299
image_sizeCols = 96#192#299
class_numMelanoma = 2
img_channelsMelanoma = 3
ithAug=0

class BatchReader:
    def __init__(self, aImgsDir, aTrainPercent=0.9, aValidPercent=0.0, aTestPercent=0.1,
                                 aTrainBatchSize=0, aValidBatchSize=1, aTestBatchSize=1,
                                 aLabelsDir = ''):
        #overwrite the parameter, because it must be 1 (todo: allow other batch sizes for test and validation)
        aValidBatchSize = 1
        aTestBatchSize = 1
        if aTrainPercent + aValidPercent + aTestPercent > 1:
            print('aTrainPercent + aValidPercent + aTestPercent > 1, exiting.')
            sys.exit(0)

        self.nTotalImages = self.__setImagesDir(aImgsDir, aLabelsDir)

        self.trainSize = math.floor(self.nTotalImages*aTrainPercent)
        self.validSize = math.floor(self.nTotalImages*aValidPercent)
        self.testSize = math.floor(self.nTotalImages*aTestPercent)

        self.trainBatchSize = aTrainBatchSize
        self.validBatchSize = aValidBatchSize
        self.testBatchSize = aTestBatchSize

        self.resetCounters()
        self.imageLoader = None
        self.placeholder_ImageFileName = None

    def getImagesDirectory(self):
        return self.imagesDir

    def resetCounters(self, boolDataShuffle=True):
        #set initial imgPointers
        self.trainImgPointer = 0
        self.validImgPointer = self.trainSize
        self.testImgPointer = self.trainSize + self.validSize

        if boolDataShuffle:
            self.allImagesFiles, self.allImagesLabels = shuffle(self.allImagesFiles, self.allImagesLabels)
            tempFiles,tempLabels=[],[]
            newFiles, newLabels=[],[]
            tempFiles,tempLabels = shuffle(self.allImagesFiles[0:self.trainSize],self.allImagesLabels[0:self.trainSize])
            newFiles+=tempFiles
            newLabels+=tempLabels
            tempFiles,tempLabels = shuffle(self.allImagesFiles[self.trainSize:(self.trainSize+self.validSize)],self.allImagesLabels[self.trainSize:(self.trainSize+self.validSize)])
            newFiles+=tempFiles
            newLabels+=tempLabels
            tempFiles,tempLabels = shuffle(self.allImagesFiles[(self.trainSize+self.validSize):],self.allImagesLabels[(self.trainSize+self.validSize):])
            newFiles+=tempFiles
            newLabels+=tempLabels
            self.allImagesFiles = newFiles
            self.allImagesLabels = newLabels

    def getNOfIterations(self, trainValidOrTest):
        if trainValidOrTest == 'Train':
            if self.trainBatchSize == 0: return 0
            return math.floor(float(self.trainSize)/float(self.trainBatchSize))
        elif trainValidOrTest == 'Valid':
            if self.validBatchSize == 0: return 0
            return math.floor(float(self.validSize)/float(self.validBatchSize))
        else:
            if self.testBatchSize == 0: return 0
            return math.floor(float(self.testSize)/float(self.testBatchSize))

    def getNOfBatchs(self,trainValidOrTest):
        return self.getNOfIterations(trainValidOrTest)

    def __setImagesDir(self, aImagesDir, aLabelsDir, bJsonFileNecessary = True):
        self.imagesDir = aImagesDir
        if self.imagesDir[-1] != '/': self.imagesDir += '/'

        self.labelsDir = aLabelsDir
        if self.labelsDir == '': self.labelsDir = self.imagesDir
        elif self.labelsDir[-1] != '/': self.labelsDir += '/'

        self.allImagesFiles = []
        iImagesCounter = 0
        #read the files names
        listOfFiles = os.listdir(self.imagesDir)

        #shuffle the files -  this shuffling will impact which images are in each partition train, valid, test
        #random.Random(43).shuffle(listOfFiles)   #for faster hyper-parameter definition
        # for robust (longer) training - use:
        random.shuffle(listOfFiles)

        nImagesWithoutLabel = 0
        self.allImagesLabels = []
        nMalignImgs = 0
        nBenignImgs = 0

        for file in listOfFiles:
            if '.jpeg' in file:
                os.rename(self.labelsDir + file, self.labelsDir + file[:-4]+'jpg')
            if '.jpg' in file:# or 'png' in file:
                if bJsonFileNecessary:
                    #if an image file has a json corresponding file
                    jsonFileName = file[0:file.rindex('.')] + '.json'
                    if not os.path.exists(self.labelsDir + jsonFileName):
                        nImagesWithoutLabel += 1
                        print('WARNING: NO JSON FILE')
                        continue
                    jsonData = json.load(open(self.labelsDir + jsonFileName))
                    # print(jsonFileName)
                    # and if the json file has benign_malignant information
                    if 'benign_malignant' in jsonData['meta']['clinical']:
                        if jsonData['meta']['clinical']['benign_malignant'] == 'benign':
                            self.allImagesLabels.append(0)  # 0 é benigno
                            nBenignImgs+=1
                        else:
                            self.allImagesLabels.append(1)
                            nMalignImgs+=1
                        iImagesCounter += 1
                        #then we add the label to self.allImagesLabels (above), and the file to self.allImagesFiles
                        self.allImagesFiles.append(self.imagesDir+file)
                    elif 'melanocytic' in jsonData['meta']['clinical']:
                        if jsonData['meta']['clinical']['melanocytic'] == 'false':
                            self.allImagesLabels.append(0)  # 0 é benigno
                            nBenignImgs+=1
                        else:
                            self.allImagesLabels.append(1)
                            nMalignImgs+=1
                        iImagesCounter += 1
                        #then we add the label to self.allImagesLabels (above), and the file to self.allImagesFiles
                        self.allImagesFiles.append(self.imagesDir+file)
                    else:
                        print('JSON FILE',jsonFileName,'has no malignance information')
                        nImagesWithoutLabel += 1
                else:
                    self.allImagesFiles.append(self.imagesDir + file)
        print('Using images directory:', self.imagesDir)
        print('Initial number of image files: ',len(self.allImagesFiles))
        print('Number of benign images: ',nBenignImgs)
        print('Number of malign images: ',nMalignImgs)
        print('Total image files with label: ', len(self.allImagesFiles))
        print('Discarded files without label: ', nImagesWithoutLabel)
        print('--------------')
        self.malignImagesRatio = nMalignImgs/len(self.allImagesFiles)
        return len(self.allImagesFiles)

    def loadTFImage(self, filename):
        if self.imageLoader == None:
            self.placeholder_ImageFileName = tf.placeholder(tf.string)
            image_string = tf.read_file(self.placeholder_ImageFileName)
            decodedImage = tf.image.decode_jpeg(image_string, channels=3)
            self.imageLoader = tf.image.convert_image_dtype(decodedImage, tf.float32)
        with tf.Session() as sess:
            loadedImage = sess.run(self.imageLoader, feed_dict={self.placeholder_ImageFileName: filename})
        return loadedImage
#-------------------------------------------------------------------------------
    def __random_crop(self, batch, crop_shape, padding=None):
        oshape = numpy.shape(batch[0])

        if padding:
            oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            for i in range(len(batch)):
                new_batch.append(batch[i])
                if padding:
                    new_batch[i] = numpy.lib.pad(batch[i], pad_width=npad,
                                                 mode='constant', constant_values=0)
                nh = random.randint(0, oshape[0] - crop_shape[0])
                nw = random.randint(0, oshape[1] - crop_shape[1])
                new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                               nw:nw + crop_shape[1]]
            return new_batch

    def __random_flip_leftright(self, batch):
        flippedBatch = []
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                flippedBatch[i] = numpy.fliplr(batch[i])
        return flippedBatch

    def data_augmentation(self, batch):
        flippedBatch = self.__random_flip_leftright(batch)
        # batch = __random_crop(batch, [image_sizeRows, image_sizeCols])
        return flippedBatch

# -------------------------------------------------------------------------------
    def getAllImagesFiles(self, trainValidOrTest, bShuffleOrder = True):
        temp = None
        if trainValidOrTest == 'Train':
            temp = self.allImagesFiles[0:self.trainSize]
            print('Providing ', len(temp),' image file names for Train out of a total of ', len(self.allImagesFiles), ' images - number of Train iterations: ', self.getNOfBatchs('Train'),'for batch size',self.trainBatchSize)
        elif trainValidOrTest == 'Valid':
            temp = self.allImagesFiles[self.trainSize:(self.trainSize+self.validSize)]
            print('Providing ', len(temp),' image file names for Validation out of a total of ', len(self.allImagesFiles), ' images - number of Train iterations: ', self.getNOfBatchs('Valid'),'for batch size',self.validBatchSize)
        else:
            temp = self.allImagesFiles[(self.trainSize+self.validSize):]
            print('Providing ', len(temp),' image file names for Test out of a total of ', len(self.allImagesFiles), ' images - number of Train iterations: ', self.getNOfBatchs('Test'),'for batch size',self.testBatchSize)
        if bShuffleOrder:
            shuffle(temp)   #shuffles only the order, not the  elements of the set - impacts the batches, different each time
        return temp

    def getAllImagesLabels(self, trainValidOrTest):
        temp = None
        if trainValidOrTest == 'Train':
            temp = self.allImagesLabels[0:self.trainSize]
        elif trainValidOrTest == 'Valid':
            temp = self.allImagesLabels[self.trainSize:(self.trainSize+self.validSize)]
        else:
            temp = self.allImagesLabels[(self.trainSize+self.validSize):]
        return temp
# -------------------------------------------------------------------------------
    def getNextBatch(self, trainValidOrTest, augmentationTimes = 0, bConvertLabelsToHotVectors = True):
        #returns decoded images instead of just their file names
        #return labels as hot-vectors instead of scalar values
        #->we use this for testing, since it is not using a data pipeline
        if trainValidOrTest == 'Train':
            imgPointer, batchSize = self.trainImgPointer, self.trainBatchSize
        elif trainValidOrTest == 'Valid':
            imgPointer, batchSize = self.validImgPointer, self.validBatchSize
        else:
            imgPointer, batchSize = self.testImgPointer, self.testBatchSize
        nextImgPointer = imgPointer + batchSize

        imgBatch = []
        labelBatch = []

        for ithImg in range(imgPointer, nextImgPointer):
            img = self.loadTFImage(self.allImagesFiles[ithImg])
            if img.shape[0] != image_sizeRows or img.shape[1] != image_sizeCols:
                print("ERROR: Verify image resolution: ", img.shape, " expected ", image_sizeRows, " x ", image_sizeCols)
            imgBatch.append(img)
            labelBatch.append(self.allImagesLabels[ithImg])
            # print(data.shape)
        # final tensor must be #images x #image_rows x #image_columns x #color_channels = #images x 96 x 96 x 3
        # converts list of #image_rows x #image_columns x #color_channels to array #images x #image_rows x #image_columns x #color_channels
        imgBatch = numpy.array(imgBatch)
        labelBatch = numpy.array(labelBatch)
        #print(imgBatch.shape)
        #print(labelBatch.shape)

        #convert labels to vectors: 0->[1,0]; 1->[0,1]
        if bConvertLabelsToHotVectors:
            labelBatch = numpy.array([[float(i == label) for i in range(class_numMelanoma)] for label in labelBatch])

        for aug in range(augmentationTimes):
            #duplicates the entire batch
            augmentedBatch = self.data_augmentation(imgBatch)
            #concatenates the new data to the batch
            imgBatch += augmentedBatch
            labelBatch += labelBatch

        if trainValidOrTest == 'Train':
            self.trainImgPointer = nextImgPointer
        elif trainValidOrTest == 'Valid':
            self.validImgPointer = nextImgPointer
        else:
            self.testImgPointer = nextImgPointer

        return imgBatch, labelBatch

    def getNextRawBatch(self, trainValidOrTest, augmentationTimes = 0):
        #returns images file names instead of decoded images
        #return labels as scalar values instead of hot-vectors
        if trainValidOrTest == 'Train':
            imgPointer, batchSize = self.trainImgPointer, self.trainBatchSize
        elif trainValidOrTest == 'Valid':
            imgPointer, batchSize = self.validImgPointer, self.validBatchSize
        else:
            imgPointer, batchSize = self.testImgPointer, self.testBatchSize
        nextImgPointer = imgPointer + batchSize

        imgFileNamesBatch = []
        labelBatch = []
        for ithImg in range(imgPointer, nextImgPointer):
            imgFileNamesBatch.append(self.allImagesFiles[ithImg])
            labelBatch.append(self.allImagesLabels[ithImg])
            # print(data.shape)

        if trainValidOrTest == 'Train':
            self.trainImgPointer = nextImgPointer
        elif trainValidOrTest == 'Valid':
            self.validImgPointer = nextImgPointer
        else:
            self.testImgPointer = nextImgPointer

        return imgFileNamesBatch, labelBatch

    def performTFAugmentationOfMalignIamgesOnDisk(self, image, label):
        global ithAug
        fileName = image
        if tf.equal(label,tf.constant(1)):
            print_out = tf.Print(image, [image], "Malign input file: ")
            image = print_out
            image_string = tf.io.read_file(image)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            #image = tf.image.resize_images(image, [image_sizeRows, image_sizeCols])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.clip_by_value(image, 0.0, 1.0)
            image = tf.image.convert_image_dtype(image, tf.uint8)
            image = tf.image.encode_jpeg(image)
            newFileName = tf.strings.join([tf.strings.substr(fileName,0,tf.strings.length(fileName)-4),"_"+str(ithAug)+".jpg"])
            tf.io.write_file(newFileName,image)
            jsonFileName = tf.strings.join([tf.strings.substr(fileName, 0, tf.strings.length(fileName) - 4), ".json"])
            newJasonFileName = tf.strings.join([tf.strings.substr(fileName, 0, tf.strings.length(fileName) - 4), "_" + str(ithAug) + ".json"])
            json_string = tf.io.read_file(jsonFileName)
            tf.io.write_file(newJasonFileName,json_string)
        return image, label

    def resizingOnDisk(self, image, label):
        image = tf.cast(image,tf.string)
        fileName = image
        image = tf.Print(image, [image], "Input file: ")
        image_string = tf.io.read_file(image)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [image_sizeRows, image_sizeCols])
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.encode_jpeg(image)
        newFileName = tf.strings.join([tf.strings.substr(fileName,0,tf.strings.length(fileName)-4),"_resized.jpg"])
        print_out = tf.Print(newFileName, [newFileName], "Output file: ")
        newFileName = print_out
        tf.io.write_file(newFileName,image)
        jsonFileName = tf.strings.join([tf.strings.substr(fileName, 0, tf.strings.length(fileName) - 4), ".json"])
        newJasonFileName = tf.strings.join([tf.strings.substr(fileName, 0, tf.strings.length(fileName) - 4), "_resized.json"])
        json_string = tf.io.read_file(jsonFileName)
        tf.io.write_file(newJasonFileName, json_string)
        return image, label

def resizeImages(directoryPath):
    if '15' not in tf.__version__:
        print('ERROR: Tensor flow version 1.15 is required. Quiting.')
        sys.exit(0)
    batchReader = BatchReader(directoryPath) #works only if the image files have corresponding json files
    sys.exit(0)
    #The original images keep untouched, it is necessary to go to the specified directory and copy the files with _resized naming
    #to the new directory of files
    dataset = tf.data.Dataset.from_tensor_slices((batchReader.allImagesFiles, batchReader.allImagesLabels))
    dataset = dataset.map(batchReader.resizingOnDisk)
    dataset = dataset.repeat()
    dataset = dataset.batch(1)
    iter = dataset.make_one_shot_iterator()
    x, y = iter.get_next()
    input_x = x + x #nothing, but a node in the computing graph
    with tf.Session() as sess:
        for  i in range(len(batchReader.allImagesFiles)):
            if i % 1000==0: print(str(i), '.000 images resized')
            temp = sess.run(input_x)

def augmentImagesOnDisk(directoryPath, augmTimes = 4):
    if '15' not in tf.__version__:
        print('ERROR: Tensor flow version 1.15 is required. Quiting.')
        sys.exit(0)
    global ithAug
    batchReader = BatchReader(directoryPath)

    #The original images keep untouched, it is necessary to go to the specified directory
    # and copy the files with _X naming to the new directory of files
    for i in range(augmTimes):
        ithAug = i
        dataset = tf.data.Dataset.from_tensor_slices((batchReader.allImagesFiles, batchReader.allImagesLabels))
        dataset = dataset.map(batchReader.performTFAugmentationOfMalignIamgesOnDisk)
        dataset = dataset.repeat()
        dataset = dataset.batch(1)
        iter = dataset.make_one_shot_iterator()
        x, y = iter.get_next()
        input_x = x + x
        with tf.Session() as sess:
            for  i in range(len(batchReader.allImagesFiles)):
                temp = sess.run(input_x)

def convertEntireDirectoryToCLAHE(directoryPath):
    if directoryPath[-1] != '/': directoryPath += '/'
    listOfFiles = os.listdir(directoryPath)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    i = 0
    for file in listOfFiles:
        if '.jpg' in file:
            img = cv2.imread(directoryPath+file, 0)
            cl = clahe.apply(img)
            cv2.imwrite(directoryPath+file,cl)
            i += 1
            if i % 50 == 0: print(str(i),"files processed")

def checkDir(directoryPath):
    if '15' not in tf.__version__:
        print('ERROR: Tensor flow version 1.15 is required. Quiting.')
        sys.exit(0)
    batchReader = BatchReader(directoryPath)

if __name__ == '__main__':
    #check a given directory by default
    #batchReader = BatchReader('./ALLMIXED-CROPPED-CLASHED')
    #augmentImagesOnDisk('./ALLMIXED-CROPPED-CLASHED')
    #resizeImages('/home/junio/Desktop/Ju/MelanomaClassification/TEST/POINT-7-CROPPED-CLASHED/')
    resizeImages('/home/junio/Desktop/Ju/MelanomaClassification/ISIC-23902/train-18906')
    #convertEntireDirectoryToCLAHE('/home/junio/Desktop/Ju/MelanomaClassification/TEST/HAM10000_96x9-CLASHED/')
