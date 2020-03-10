# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# less verbosity, in GPU: PYTHONWARNINGS=ignore python SE_ResNeXt.py
# in CPU: PYTHONWARNINGS=ignore TF_XLA_FLAGS=--tf_xla_cpu_global_jit python SE_ResNeXt.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
from keras.constraints import maxnorm
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from sklearn import metrics
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tflearn.layers.conv
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import numpy as np
from ClassBatchReader import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('xla_compilation_cache', 1, '')

momentum = 0.9
regularizationParameter = 0.0005
init_learning_rate = 0.1

bDebugDimensionality = False
if (not tf.test.is_gpu_available()):
    print('=> No GPU, using CPU mode.')
    CPU_GPU = 'CPU'
    bDebugDimensionality = True
else:
    print('=> GPU detected, using Cuda.')
    CPU_GPU = 'GPU'

settings = {}
if CPU_GPU == 'CPU':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    settings = {'cardinality': 2,
                'cardinality_of_residual_block': 1,
                'split_out_channels': 2,
                'reduction_ratio': 16,
                'total_epochs': 2,
                'imagesDir': './ISIC2017CLASHED/train/',
                'testImagesDir': './TEST/PH2_96x96/',
                'validImagesDir': './ISIC/CROPPED-CLASHED/test',
                'bKernelConstraint': False,
                'bTestOnly':False,
                'loss': 'hinge_loss',  # hinge_loss or cross_entropy
                'train_batch_size': 1,
                'num_of_grad_iterations': 100,
                'comments': 'trying split_out > cardinality'}
else:
    settings = {'cardinality': 8,
                'cardinality_of_residual_block': 3,
                'split_out_channels': 8,
                'reduction_ratio': 2,
                'total_epochs': 100,
                'imagesDir': './ISIC-23906/train-18906-cropped-resized',
                'testImagesDir': './ISIC-23906/test-2500-cropped-resized',
                'validImagesDir': './ISIC-23906/valid-2500-cropped-resized',
                'bKernelConstraint': True,
                'bTestOnly': False,
                'loss': 'hinge_loss',  # hinge_loss or cross_entropy
                'train_batch_size': 100,
                'num_of_grad_iterations': 1,
                'comments': 'minimal architecture'}

settingsString = 'CrossEntropy-ISIC23906' + str(settings['cardinality']) + '-b' \
                 + str(settings['cardinality_of_residual_block']) + '-s' + str(settings['split_out_channels']) \
                 + '-r' + str(settings['reduction_ratio']) + 'KernelConstraint'
print('=> SETTINGS: ',settingsString)

checkpointDir = './checkpoints/' + settingsString + '/'
if not os.path.exists('./checkpoints/'):
    os.mkdir('./checkpoints/')
if not os.path.exists(checkpointDir):
    os.mkdir(checkpointDir)

bestValidationModelDirectory = './models/melanoma_prediction/best-' + settingsString + '/'
if os.path.exists(bestValidationModelDirectory + 'bestValidationAccuracy.txt'):
    with open(bestValidationModelDirectory + 'bestValidationAccuracy.txt', 'r') as temp:
        bestValidationAccuracy = float(temp.readline().strip())
else:
    bestValidationAccuracy = 0.0


class DermaDL():
    def __init__(self, x, training):
        self.training = training
        self.model = self.BuildNeuralNetwork(x)

    def convolution(self, input, filter, kernel, stride, classifier, padding='SAME', layer_name="conv"):
        # CNNs arrange its tensors in four dimensions (batch_size, height, width, channels)
        if bDebugDimensionality: input = tf.Print(input, [tf.shape(input), tf.shape(input)[3], kernel],
                                                  'Before convolution ' + layer_name + ' with given kernel: ')
        with tf.name_scope(layer_name):
            if settings['bKernelConstraint']:
                network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel,
                                           strides=stride, padding=padding, kernel_constraint=maxnorm(3.0))
            else:
                network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel,
                                           strides=stride, padding=padding)
            # this does not help at all
            # network = tf.cond(classifier.training, lambda: dropblock(network, 0.5, 5), lambda: network)

            # after the convolution, the number of channels will always be equal parameter 'filter'
            if bDebugDimensionality: network = tf.Print(network, [tf.shape(network), tf.shape(network)[3], kernel],
                                                        'After convolution ' + layer_name + ' with given kernel: ')
            return network

    def batchNormalization(self, x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True):
            return tf.cond(training,
                           lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda: batch_norm(inputs=x, is_training=training, reuse=True))

    def initialConvolution(self, x, filter, scope):
        # the first layer adds an initial convolution operation
        # x: (batch_size, height, width, channels); initially, channels = 3, as for RGB
        if bDebugDimensionality: x = tf.Print(x, [tf.shape(x), tf.shape(x)[3]], 'Before initial_convolution: ')
        with tf.name_scope(scope):
            x = self.convolution(input=x, filter=filter, kernel=[3, 3], stride=1, classifier=self, layer_name=scope + '_conv1')
            x = self.batchNormalization(x, training=self.training, scope=scope + '_batch1')
            x = tf.nn.relu(x)
            # output: x: (batch_size, height, width, channels=filter)
            if bDebugDimensionality: x = tf.Print(x, [tf.shape(x), tf.shape(x)[3]], 'After initial_convolution: ')
            return x

    def splitLayer(self, input_x, stride, layer_name):
        if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'Before split_layer: ' + layer_name + ": ")
        with tf.name_scope(layer_name):
            layers_split = list()
            # read about spliting and cardinality at https://arxiv.org/pdf/1611.05431.pdf
            for i in range(settings['cardinality']):
                # each time of this iteration, a (batch_size,height,width,depth) tensor is created
                # depth = settings['split_out_channels']
                # at the end, there will be a list with 'cardinality'*(batch_size,height,width,depth)-tensors
                # which will be concatenated into a (batch_size,height,width,depth*cardinality)-tensor
                split = input_x
                scope = layer_name + '_splitN_' + str(i)
                with tf.name_scope(scope):
                    # 1x1 convolutin are used to alter the number of channels; read about at https://iamaaditya.github.io/2016/03/one-by-one-convolution/ - Fig 3(c)
                    # briefly, a total of "filter 1x1 convolutions" operate, each, over all the input channels; the weights make it work like a pooling throughout the layers
                    if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'Before ' + scope + '_conv1: ')
                    split = self.convolution(split, filter=settings['split_out_channels'], kernel=[1, 1], stride=1, classifier=self, layer_name=scope + '_conv1')
                    split = self.batchNormalization(split, training=self.training, scope=scope + '_batch1')
                    split = tf.nn.relu(split)
                    if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'after ' + scope + '_conv1: ')
                    if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'before ' + scope + '_conv2: ')
                    # adds a 3x3 convolution as well - this convolution will change the sizes of dimensions height and width
                    # the new dim, dim_after3x3conv=(dim-filter+2*padding)/stride+1 (the same for height or width)
                    # for example height=(96-3+2*2)/2 + 1 = 48
                    # this dimensionality will flow until the end of the network - everytime the split layer is evolked it reduces even more
                    split = self.convolution(split, filter=settings['split_out_channels'], kernel=[3, 3], stride=stride, classifier=self, layer_name=scope + '_conv2')
                    split = self.batchNormalization(split, training=self.training, scope=scope + '_batch2')
                    split = tf.nn.relu(split)
                    if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'after ' + scope + '_conv2: ')
                layers_split.append(split)
            # concatenates all the "cardinality" tensors into a (depth, height, width, channels="cardinality" x depth) tensor
            # for example, if cardinality = 16 and depth=64 the result is 16 x 64 = 1024
            if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'Before all the ' + str(settings['cardinality']) + ' split_layer concatenation ' + layer_name + ': ')
            input_x = tf.concat(layers_split, axis=3)
            if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'After all the ' + str(settings['cardinality']) + ' split_layer concatenation ' + layer_name + ': ')
            return input_x

    def fullyConnected(self, x, units, layer_name='fullyConnected'):
        with tf.name_scope(layer_name):
            x = tf.layers.dense(inputs=x, use_bias=False, units=units)
            x = tf.layers.dropout(inputs=x, rate=0.5)
            return x
        
    def squeezeAndExcitation(self, input_x, out_dim, layer_name):
        # input_x: (batch_size, height, width, channels=out_dim)
        with tf.name_scope(layer_name):
            # read about the squeeze-excitation technique at
            # http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf

            # global_avg_pool: each channel is reduced to one single value, the avg of all its values (channel-wise statistics)
            # returns a vector with rank=depth (one avg for each channel)- so the name, squeeze -> section 3.1 of the paper
            if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'Before squeeze at layer ' + layer_name)
            squeeze = tflearn.layers.conv.global_avg_pool(input_x, name='Global_avg_pooling')
            if bDebugDimensionality: squeeze = tf.Print(squeeze, [tf.shape(squeeze)], 'After squeeze at layer ' + layer_name)
            # squeeze: (batch_size, channels=out_dim)

            # excitation -> section 3.2 of the paper
            # fully-connected layer that alter the dimensionality and apply non-linearities: relu than sigmod
            excitation = self.fullyConnected(squeeze, units=out_dim / settings['reduction_ratio'], layer_name=layer_name + '_fullyConnected1')
            excitation = tf.nn.relu(excitation)
            if bDebugDimensionality: excitation = tf.Print(excitation, [tf.shape(excitation)], 'After applying excitation reduction ration at layer ' + layer_name + ': ')
            # excitation: (batch_size, channels=out_dim/reduction_ratio)

            # restore dimensionality to out_dim with a fully-connected layer
            excitation = self.fullyConnected(excitation, units=out_dim, layer_name=layer_name + '_fullyConnected2')
            excitation = tf.nn.sigmoid(excitation)
            # excitation: (batch_size, channels=out_dim)
            if bDebugDimensionality: excitation = tf.Print(excitation, [tf.shape(excitation)], 'After excitation at layer ' + layer_name + ': ')

            # reshapes the squeezed matrix (batch_size, channels=out_dim) to tensor (batch_size x 1 x 1 x channels=outdim)
            # in function reshape, the -1 indicates that the rank of this dimension can alter as needed to fit all the elements
            # that is, it simply embbeds the vector into a 4d tensor, whose rank of dimension 3 is out_dim, the same as of tensor input_x
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            if bDebugDimensionality: excitation = tf.Print(excitation, [tf.shape(excitation)], 'After excitation reshape at layer ' + layer_name + ': ')

            # this line characterizes the skip nature of the squeeze-excitation method
            # the original input_x operates over the result of the squeeze-excitation
            # the element-wise multiplication corresponds to
            # (batch_size, height, width, out_dim) * (1x1x1xout_dim)
            scale = input_x * excitation
            # scale: (batch_size, height, width, channels=out_dim)
            if bDebugDimensionality: scale = tf.Print(scale, [tf.shape(scale)], 'After SE final scale at layer ' + layer_name + ': ')
            return scale

    def residualBlock(self, input_x, out_dim, layer_num):
        # input_x: (batch_size, height, width, channels="filter as defined in the initial_convolution")
        if bDebugDimensionality: input_x = tf.Print(input_x, [tf.shape(input_x), tf.shape(input_x)[3]], 'Before residual_block: ')

        # for each residual block in a total of "settings['cardinality_of_residual_block']"
        for i in range(settings['cardinality_of_residual_block']):
            # input_n_of_channels is the rank of the last dimension of input_x
            # this snnipet takes care of cardinality compliance of the skip connection
            input_n_of_channels = int(np.shape(input_x)[-1])
            if input_n_of_channels * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_n_of_channels // 2
            else:
                flag = False
                stride = 1

            # read about spliting at https://arxiv.org/pdf/1611.05431.pdf - splits the tensor, then concatenates it
            # input_x: (batch_size, height, width, channels="filter as defined in the initial_convolution")
            x = self.splitLayer(input_x, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
            # the split_layer performs a 3x3 convolution which reduces the sizes of height and width
            # ==> each time it is evoked, height and width reduces a little more
            # x: (batch_size, height, width, channels=cardinality x depth) -> after the split, the rank of dimension 3 is very high
            # depth = settings['split_out_channels']

            # a small 1x1 convolution to reduce channels from "cardinality x depth" to "out_dim"
            x = self.convolution(x, filter=out_dim, kernel=[1, 1], stride=1, classifier=self, layer_name='trans_layer_' + layer_num + '_' + str(i))
            x = self.batchNormalization(x, training=self.training, scope='trans_layer_' + layer_num + '_' + str(i) + '_batch1')
            x = tf.nn.relu(x)
            # x: (batch_size, height, width, channels=out_dim)

            # Squeeze and excitation: http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
            x = self.squeezeAndExcitation(x, out_dim=out_dim, layer_name='squeeze_layer_' + layer_num + '_' + str(i))

            # since we are using a skip connection (residual), after the residual block, the dimensionality
            # of input_x must be the same as of the tensor x, after the block processing
            # these lines here make the appropriate padding to make sure the dimensionality is ok
            # (batch_size, height, width, channels="filter as defined in the initial_convolution") => (batch_size, height, width, channels=out_dim)
            if flag is True:
                pad_input_x = tf.layers.average_pooling2d(inputs=input_x, pool_size=[2, 2], strides=2, padding='SAME')
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = input_x

            # this line characterizes a residual connection, just as of resnet: https://arxiv.org/abs/1512.03385
            # that is, the original input is added to the result of the processing of the block (beggining of the procedure)
            residual_block_output_x = tf.nn.relu(x + pad_input_x)

        if bDebugDimensionality: residual_block_output_x = tf.Print(residual_block_output_x, [tf.shape(residual_block_output_x), tf.shape(residual_block_output_x)[3]], 'After residual_block: ')
        # residual_block_output_x: (batch_size, height, width, channels=out_dim)
        return residual_block_output_x


    def BuildNeuralNetwork(self, input_x):
        '''Number of layers
         +1 initial convolution
         +(3)*(settings['cardinality_of_residual_block']) #each resisual_block: split + convolution + squeeze_excite=3
         +1 fully connected'''

        # x: (batch_size, height, width, channels=3)
        x = self.initialConvolution(input_x, filter=8, scope='initial_convolution')
        # x: (batch_size, height, width, channels=8)
        x = self.residualBlock(x, out_dim=8, layer_num='1')
        # x: (batch_size, height, width, channels=8)
        x = self.residualBlock(x, out_dim=16, layer_num='2')
        # x: (batch_size, height, width, channels=16)
        x = self.residualBlock(x, out_dim=32, layer_num='3')
        # x: (batch_size, height, width, channels=32)

        if bDebugDimensionality: x = tf.Print(x, [tf.shape(x), tf.shape(x)[3]], 'Before global_avg_pool: ')
        # for each channel, computes the global average
        x = tflearn.layers.conv.global_avg_pool(x, name='Global_avg_pooling')
        # x: (batch_size, channels=256)
        if bDebugDimensionality: x = tf.Print(x, [tf.shape(x)], 'After global_avg_pool: ')

        # compute the logits with output = class_numMelanoma; later, output will pass through softmax
        x = self.fullyConnected(x, units=class_numMelanoma, layer_name='final_fullyConnected')
        # x: (batch_size, channels=units=2)
        if bDebugDimensionality: x = tf.Print(x, [tf.shape(x)], 'After final fullyConnected: ')
        return x


def fetchImageFile(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
def testSavedModel(batchReader, savedModelDirectory=None):
    accurate, wrong, benign, malign = 0, 0, 0, 0
    # https://stackoverflow.com/questions/45705070/how-to-load-and-use-a-saved-model-on-tensorflow
    fullListOfTrueYOutcomeForAUCROCAndPR_list = []
    fullListOfPredictedYProbsForAUCROC_list = []
    fullListOfPredictedYForPrecisionRecall_list = []
    if savedModelDirectory is None: savedModelDirectory = bestValidationModelDirectory
    with tf.Session(graph=tf.Graph()) as sess:
        metaGraphDef = tf.saved_model.loader.load(sess=sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=savedModelDirectory)
        print('=>Model loaded from', savedModelDirectory)
        # print(metaGraphDef.signature_def['melanoma_prediction'].inputs)
        # print(metaGraphDef.signature_def['melanoma_prediction'].outputs)
        graph = tf.get_default_graph()

        # print(graph.get_operations())
        placeholder_X = graph.get_tensor_by_name("IteratorGetNext:0")
        model = graph.get_tensor_by_name("Softmax:0")
        training_flag = graph.get_tensor_by_name("Placeholder_2:0")

        test_iterations = batchReader.getNOfIterations('Test')
        fp, tp, fn, tn = 0, 0, 0, 0
        for ith in range(test_iterations):
            ithTestInstance, ithTestLabel = batchReader.getNextBatch('Test', bConvertLabelsToHotVectors=False)
            # ithTestInstance is an image in tensor format; ithTestLabel is a one-value vector [ith_label]
            # print("Label:",test_y)

            predicted = sess.run(model, feed_dict={placeholder_X: ithTestInstance,
                                                   training_flag: False})  # returns two values [prob_0, prob_1]
            # since the batch size for test is 1, the index is 0 for both ithTestInstance and ithTestLabel
            # ithTestLabel is a scalar with the value of the class, it is not a vector
            if (ithTestLabel[0] == 0):
                benign += 1
            else:
                malign += 1

            if (np.argmax(predicted) == ithTestLabel[0]):
                accurate += 1
                if (ithTestLabel[0] == 0):
                    tn += 1
                    print('Correct prediction for ',str(ith)+'-th test sample: True negative')
                else:
                    tp += 1
                    print('Correct prediction for ',str(ith)+'-th test sample: True positive')
            else:
                wrong += 1
                if (ithTestLabel[0] == 0):
                    fp += 1
                    print('Wrong prediction for ',str(ith)+'-th test sample: False positive')
                else:
                    fn += 1
                    print('Wrong prediction for ',str(ith)+'-th test sample: False negative')

            fullListOfTrueYOutcomeForAUCROCAndPR_list.append(ithTestLabel[0])
            fullListOfPredictedYProbsForAUCROC_list.append(predicted[0][1])  # score of the class with greater label, that is, class 1
            fullListOfPredictedYForPrecisionRecall_list.append(np.argmax(predicted)) # label of the class with greater prob

    print("Total correct: ", accurate, " out of ", test_iterations, " => ", "{0:.2f}".format(accurate / test_iterations * 100), '%')
    print("Total incorrect: ", wrong)
    print("Benign: ", benign, " - malign: ", malign)
    print('CONFUSION MATRIX')
    print('True negative..: ', tn, 'out of', benign, '=>', "{0:.2f}".format(tn / benign * 100), '%')
    print('False positive.: ', fp, 'out of', benign, '=>', "{0:.2f}".format(fp / benign * 100), '%')
    print('True positive..: ', tp, 'out of', malign, '=>', "{0:.2f}".format(tp / malign * 100), '%')
    print('False negative.: ', fn, 'out of', malign, '=>', "{0:.2f}".format(fn / malign * 100), '%')

    print(fullListOfTrueYOutcomeForAUCROCAndPR_list)
    print(fullListOfPredictedYProbsForAUCROC_list)
    print(fullListOfPredictedYForPrecisionRecall_list)
    # ------------ROC AUC and Precision Recall
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    print("Weighted AUC-ROC score: " + "{0:.2f}".format(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
                                                                              fullListOfPredictedYProbsForAUCROC_list,
                                                                              average='weighted')))
    print("Macro AUC-ROC score: " + "{0:.2f}".format(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
                                                                              fullListOfPredictedYProbsForAUCROC_list,
                                                                              average='macro')))
    print('SENSITIVITY: ',str(tp/(tp+fn)))
    print('SPECIFICITY: ',str(tn/(tn+fp)))
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,
                                                        fullListOfPredictedYForPrecisionRecall_list)
    print('Precision: {}'.format(PRResults[0]))
    print('Recall: {}'.format(PRResults[1]))
    print('Binary F1 Score: {}'.format(PRResults[2]))  # FBeta score with beta = 1.0
    print('Support: {}'.format(PRResults[3]))
    print(10 * '---')


def performValidation(sess, batchReader, valid_iterator):
    valid_acc, valid_loss = 0.0, 0.0
    valid_iterations = batchReader.getNOfBatchs('Valid')
    sess.run(valid_iterator, feed_dict={placeholder_X: batchReader.getAllImagesFiles('Valid'),
                                        placeholder_Y: batchReader.getAllImagesLabels('Valid')})
    for step in range(valid_iterations):
        lossValid, accValid = sess.run([cost, accuracy], feed_dict={training_flag: False})
        valid_acc += accValid
        valid_loss += lossValid
    # avg accumulators
    valid_loss /= valid_iterations  # average loss
    valid_acc /= valid_iterations  # average accuracy
    valid_summary = tf.Summary(value=[tf.Summary.Value(tag='valid_loss', simple_value=valid_loss),
                                      tf.Summary.Value(tag='valid_accuracy', simple_value=valid_acc)])
    return valid_acc, valid_loss, valid_summary


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
print('Scanning image data')
if settings['bTestOnly']:
    # in case I do not want to train, but only test over the test partition
    # testSavedModel(batchReader)

    batchReader = BatchReader(settings['testImagesDir'], 0.0, 0.0, 1)
    testSavedModel(batchReader)
    batchReader = BatchReader('./TEST/CROPPED-CLASHED/POINT-7-CROPPED-CLASHED', 0.0, 0.0, 1)
    testSavedModel(batchReader)
    batchReader = BatchReader('./TEST/CROPPED-CLASHED/PH2-CROPPED-CLASHED', 0.0, 0.0, 1)
    testSavedModel(batchReader)

    sys.exit(0)
else:
    if CPU_GPU == 'CPU':
        batchReader = BatchReader(settings['imagesDir'], 0.9, 0.2, 0.0, settings['train_batch_size'])
    else:
        batchReader = BatchReader(settings['imagesDir'], 1.0, 0.0, 0.0, settings['train_batch_size'])

# using the gradient averaging solution provided at: https://gchlebus.github.io/2018/06/05/gradient-averaging.html#fn:1
# more about the technique: https://bitbucket.org/limadm/tf-jpegtest/
# https://devblogs.nvidia.com/fast-ai-data-preprocessing-with-nvidia-dali/
# notice: due to the use of train_iterator, each iteration processess batchReader.trainBatchSize images at once
# hence, after numOfGradientIterations iterations, we will have processed numOfGradientIterations*batchReader.trainBatchSize
# that is, by using the gradient averaging, the actual size of the batch will be batchSize*numOfGradientIterations
numOfGradientIterations = settings['num_of_grad_iterations']
# for example, for batches of size 100, use batch size 25 and numOfGradientIterations 4

print('Train images directory:',batchReader.getImagesDirectory())
print('Number of images for training: ', batchReader.trainSize, '; Batch size: ', batchReader.trainBatchSize)
print('Number of gradient iterations:', numOfGradientIterations)
print('Actual batch size (gradient iterations * batch size): ', numOfGradientIterations * batchReader.trainBatchSize)
print()
print()
# ----------------------------------------------------------Placeholders
# image_sizeRows, image_sizeCols, img_channelsMelanoma = 3, class_num = 2
# https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
# placeholder_X = tf.placeholder(tf.float32, shape=[None, image_sizeRows, image_sizeCols, img_channelsMelanoma]) #None indicates that mini-batches are coming
# placeholder_Y = tf.placeholder(tf.float32, shape=[None, class_numMelanoma])
placeholder_X = tf.placeholder(tf.string, shape=[None])  # None indicates that mini-batches are coming - a list of image files
placeholder_Y = tf.placeholder(tf.int32, shape=[None])  # A list of scalars 0 or 1
training_flag = tf.placeholder(tf.bool)

valid_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_Y))
#valid_dataset = valid_dataset.shuffle(buffer_size=batchReader.nTotalImages, seed=43)
valid_dataset = valid_dataset.shuffle(buffer_size=batchReader.nTotalImages,reshuffle_each_iteration=True)
valid_dataset = valid_dataset.map(fetchImageFile, num_parallel_calls=4)
valid_dataset = valid_dataset.repeat()  # will repeat as many times as required by a sess.run call; if provide a number, will limit the number of times
valid_dataset = valid_dataset.batch(batchReader.validBatchSize)
valid_dataset = valid_dataset.prefetch(1)

train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_Y))
#train_dataset = train_dataset.shuffle(buffer_size=batchReader.nTotalImages, seed=43)  # for faster hyper-parameter definition
#train_dataset = train_dataset.shuffle(buffer_size=batchReader.nTotalImages)
# for robust (longer) training - use:
train_dataset = train_dataset.shuffle(buffer_size=batchReader.nTotalImages,reshuffle_each_iteration=True)
train_dataset = train_dataset.map(fetchImageFile, num_parallel_calls=4)
train_dataset = train_dataset.repeat()  # will repeat as many times as required by a sess.run call; if provide a number, will limit the number of times
train_dataset = train_dataset.batch(batchReader.trainBatchSize)
train_dataset = train_dataset.prefetch(1)

iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
data_x, data_y = iter.get_next()

valid_iterator = iter.make_initializer(valid_dataset)
train_iterator = iter.make_initializer(train_dataset)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
# ----------------------------------------------------------Core processing
# instance of the network with placeholder data_x
# logits is a term which refers to the last output vector of the network before normalization
logits = DermaDL(data_x, training=training_flag).model
# logits = tf.Print(logits,[logits],'LOGITGS: ')
# data_y = tf.Print(data_y,[data_y],'data_y: ')
# performs softmax normalization, then computes the cross_entropy cost considering the correct "labels" and the output "logits"
# y_pred = tf.arg_max(logits,1)
y_hot_vector = tf.one_hot(indices=data_y, depth=2)

if settings['loss'] == 'hinge_loss':
    cost = tf.losses.hinge_loss(labels=y_hot_vector,logits=logits,weights=[[0.2,0.7]]) #weights peanlyze false negative
else:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_hot_vector, logits=logits))

# performs softmax normalization (for testing only) - it returns a 2-dimensional vector for each image
prediction = tf.nn.softmax(logits=logits)  # this end computing node is used for prediction on production
# sums l2 of all the trainable tf.variable(trainable=True)
# add_n waits for all of its inputs to be ready
# used for l2 regularization
l2_norm = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
# instantiates Momentum Optimizer; use_nesterov=True makes it use the Nesterov version of the algorithm
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
number_of_weights = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print('The number of weights for this model is: ',sess.run(number_of_weights))

# using the gradient averaging solution provided at: https://gchlebus.github.io/2018/06/05/gradient-averaging.html#fn:1 (the else part)
if numOfGradientIterations == 1:  # initialize numOfGradientIterations with 1 to skip avg grad computing
    # Method 'minimize' = 'compute_gradients()' + 'apply_gradients()'
    updateWeights = optimizer.minimize(cost + regularizationParameter * l2_norm)
else:
    # here 'train_op' only applies gradients passed via placeholders stored
    # in 'grads_placeholders. The gradient computation is done with 'grad_op'.
    grads_and_vars = optimizer.compute_gradients(cost + regularizationParameter * l2_norm)  # 1st part of minimization
    avg_grads_and_vars = []
    _grad_placeholders = []
    for grad, var in grads_and_vars:
        grad_ph = tf.placeholder(grad.dtype, grad.shape)
        _grad_placeholders.append(grad_ph)
        avg_grads_and_vars.append((grad_ph, var))
    _grad_op = [g[0] for g in grads_and_vars]
    _apply_gradAvg_op = optimizer.apply_gradients(avg_grads_and_vars)  # 2nd part of minimization
    _gradients = []  # list to store gradients
# ----------------------------------------------------------Compute accuracy and save the model
# here, logits is a matrix train_batch_size data_x 2 - we could have used the output of softmax, but for "argmax", either way works
# compares vectors logits and data_y considering dimension 1;
# since both correspond to one-hotvectors, argmax returns the index of the highest value in each vector - hence, dimension 1 collapses
# a mini-batch is expected, so, after the argmax operation, dimension 1 collapses; correct_prediction becomes a vector, no longer a matrix
# data_y is a vector of scalars, so no need for argmax
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(data_y, tf.int64))
# correct_prediction = tf.Print(correct_prediction,[correct_prediction],'CORRECT PREDICTION VECTOR: ')
# computes the mean; reduce means - whatever the dimensionality of the input is, reduce_mean will return a single value
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy = tf.Print(accuracy,[accuracy],'ACCURACY OF CORRECT PREDICTION VECTOR: ')
# checkpoint creator
saver = tf.train.Saver(tf.global_variables())
#####################++++++++++++++++++++++++++++++=
# TRAINING
# allow_soft_placement allows dynamic allocation of GPU memory -> slightly bigger batches
if bDebugDimensionality:
    # if debbuging, uses one single thread to guarantee the sequential order of execution -> very slow
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True)
else:
    config = tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(checkpointDir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print()
        print('CHECK POINT RESTORED - resuming train for model ', checkpointDir)
        print('Notice, however, that the data was randomized before training/validation/testing partitioning, so the model needs to converge again (faster this time).')
        print()
    else:
        sess.run(tf.global_variables_initializer())
    # https://www.tensorflow.org/api_docs/python/tf/summary - optional data to be used later with tensorboard
    # summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    epoch_learning_rate = init_learning_rate
    print('Training started')
    for epoch in range(1, settings['total_epochs'] + 1):
        if epoch % (math.floor(settings['total_epochs']/3)) == 0:
            epoch_learning_rate = epoch_learning_rate / 10
        # -----------------------------------------------------------------------
        # TRAINING
        print('Epoch ', epoch)
        #after this line. train_iterator will provide batchReader.trainBatchSize at a time for training via the placeholders
        sess.run(train_iterator, feed_dict={placeholder_X: batchReader.getAllImagesFiles('Train'),
                                            placeholder_Y: batchReader.getAllImagesLabels('Train')})
        train_acc, train_loss, accuracyIterationsCounter, iterationLossCounter, gradIterationLossCounter = 0.0, 0.0, 0, 0, 0
        gradIterationLoss = 0
        train_iterations = batchReader.getNOfBatchs('Train')

        #each iteration processess batchReader.trainBatchSize images per run
        for iteration in range(train_iterations):
            train_feed_dict = {learning_rate: epoch_learning_rate, training_flag: True}
            # using the gradient averaging solution provided at: https://gchlebus.github.io/2018/06/05/gradient-averaging.html#fn:1 (the else part)
            if numOfGradientIterations == 1:
                # no grad averaging, simply runs the two graphs: cost, and updateWeights
                lossTrain, _ = sess.run([cost, updateWeights], feed_dict=train_feed_dict)
                train_loss += lossTrain
                accTrain = accuracy.eval(feed_dict=train_feed_dict)
                train_acc += accTrain
                # counters
                accuracyIterationsCounter, iterationLossCounter = accuracyIterationsCounter + 1, iterationLossCounter + 1
            else:
                lossTrain, grads = sess.run([cost, _grad_op], feed_dict=train_feed_dict)  # compute gradients, but do not apply
                gradIterationLoss += lossTrain  # loss accumulates for each gradient iteration
                gradIterationLossCounter += 1
                _gradients.append(grads)  # store the list of gradients in a list
                # compute accuracy
                accTrain = accuracy.eval(train_feed_dict)
                train_acc += accTrain
                accuracyIterationsCounter += 1
                # if reached the number of gradient iterations or recheaded the last iteration, average and apply
                # notice: due to the use of train_iterator, each iteration processess batchReader.trainBatchSize images
                # hence, when len(_gradients) == numOfGradientIterations, we will have processed numOfGradientIterations*batchReader.trainBatchSize
                if (len(_gradients) == numOfGradientIterations) or (iteration == train_iterations - 1):
                    for i, placeholder in enumerate(_grad_placeholders):
                        train_feed_dict[placeholder] = np.stack([g[i] for g in _gradients], axis=0).mean(axis=0)  # average
                    sess.run(_apply_gradAvg_op, feed_dict=train_feed_dict)  # apply gradient averages
                    _gradients = []  # clean list of gradients
                    # acumulate loss
                    train_loss += gradIterationLoss
                    iterationLossCounter += gradIterationLossCounter
                    gradIterationLoss, gradIterationLossCounter = 0, 0
            if iteration % 10 == 0:
                print('Iteration ', iteration, ' out of ', train_iterations, ' finished.')
        # ----------------------------------------------------------------------
        # -----------------------------------------------------------------------
        # avg accumulators
        with open('Trainings-outputs.txt', 'a') as f:
            f.write('acumulated accuracy: ' + str(train_acc) + ' iterations: ' + str(accuracyIterationsCounter))
            f.write('\n')
        train_loss /= iterationLossCounter  # average loss
        train_acc /= accuracyIterationsCounter  # average accuracy
        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
        # -----------------------------------------------------------------------
        # VALIDATION
        if settings['validImagesDir'] != '':
            validBatchReader = BatchReader(settings['validImagesDir'], 0.0, 1.0, 0.0)
        else:
            validBatchReader = batchReader

        if validBatchReader.getNOfIterations('Valid') > 0:
            valid_acc, valid_loss, valid_summary = performValidation(sess, validBatchReader, valid_iterator)
            # summary_writer.add_summary(summary=valid_summary, global_step=epoch)
            # summary_writer.add_summary(summary=train_summary, global_step=epoch)
            # summary_writer.flush()
            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, valid_loss: %.4f, valid_acc: %.4f \n" % (
                epoch, settings['total_epochs'], train_loss, train_acc, valid_loss, valid_acc)
            print(line)

        # Write the output of each training session
        with open('Trainings-outputs.txt', 'a') as f:
            if epoch == 1:
                f.write(15 * '-')
                f.write('\n')
                f.write(settings['comments'])
                f.write('\n')
                f.write(str(settings))
                f.write('\n')
            f.write(line)
            f.write('\n')

        # checkpoint at the end of each epoch
        saver.save(sess=sess, save_path=checkpointDir + 'ResNeXt.ckpt')

        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------
        # SAVE TRAINED MODEL
        tensor_info_x = tf.saved_model.utils.build_tensor_info(data_x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(prediction)
        flag_info_training = tf.saved_model.utils.build_tensor_info(training_flag)
        melano_prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x, 'training_info': flag_info_training},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        # no validation, therefore, consider only training results
        if validBatchReader.getNOfIterations('Valid') == 0:
            print('No validation, proceeding with training accuracy only')
            valid_acc = train_acc
        if valid_acc > bestValidationAccuracy:
            print('=> Accuracy improvement detected, saving model at ', bestValidationModelDirectory)
            print()

            # saves the best validated model, it allows for early stopping
            bestValidationAccuracy = valid_acc
            if os.path.exists(bestValidationModelDirectory):
                shutil.rmtree(bestValidationModelDirectory)
            builder = tf.saved_model.builder.SavedModelBuilder(bestValidationModelDirectory)
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={
                                                     'melanoma_prediction': melano_prediction_signature
                                                 })
            builder.save()
            with open(bestValidationModelDirectory + 'bestValidationAccuracy.txt', 'w') as temp:
                temp.write(str(bestValidationAccuracy))

# run test - it works correctly only if the batchReader is the same as the one used for training
# otherwise, there is no guarantee that train and test sets are non-intersecting
# it is also possible to create a new batchreader pointing to a directory with test images only
# batchReader = BatchReader('./TEST/PH2_96x96/', 0.0, 0.0, 1)
# testSavedModel(batchReader)
# batchReader = BatchReader('./TEST/POINT-7_96x96/', 0.0, 0.0, 1)
# testSavedModel(batchReader)
batchReader = BatchReader(settings['testImagesDir'], 0.0, 0.0, 1.0)
testSavedModel(batchReader)
