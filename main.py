import numpy as np
import pandas as pd
import random
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

magnification_list = ['40X', '100X', '200X', '400X']
benign_list = ['adenosis', 'fibroadenoma']
malignant_list = ['ductal_carcinoma', 'lobular_carcinoma']
cancer_list = benign_list + malignant_list


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_dense = labels_dense.astype(int)
    temp = index_offset + labels_dense.ravel()
    labels_one_hot.flat[temp] = 1
    return labels_one_hot


def data_split(magnification='40X', validation_percent=0.15, testing_percent=0.15, encoding='Yes'):
    validation_percent = validation_percent
    testing_percent = testing_percent
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    testing_images = []
    testing_labels = []
    for root, dirnames, filenames in os.walk("./BreakHist_Dataset/" + magnification):
        if filenames == []:
            continue
        else:
            str_length = len("./BreakHist_Dataset/40X/")
            print(root)
            if root[str_length:str_length + 6] == 'Benign':
                string_end = 31
            elif root[str_length:str_length + 9] == 'Malignant':
                string_end = 34
            elif root[str_length + 1:str_length + 7] == 'Benign':
                string_end = 32
            else:
                string_end = 35
            name = root[string_end:]
            # print(name)
            # print(cancer_list.index(name))
            total_images = 0
            for names in filenames:
                total_images += 1
            print(name, magnification, total_images)
            validation_size = np.int(total_images * validation_percent)
            testing_size = np.int(total_images * testing_percent)
            training_size = total_images - (validation_size + testing_size)
            print(training_size, validation_size, testing_size, total_images)
            num = 0

            for names in filenames:
                num += 1
                if not names.startswith('.'):
                    filepath = os.path.join(root, names)
                    print(filepath)
                else:
                    continue
                image = mpimg.imread(filepath)
                image_resize = resize(image, (115, 175), mode='constant')
                if num in range(training_size):
                    training_images.append(image_resize[:, :, :])
                    training_labels.append(cancer_list.index(name))
                elif num in range(training_size, training_size + validation_size):
                    validation_images.append(image_resize[:, :, :])
                    validation_labels.append(cancer_list.index(name))
                elif num in range(training_size + validation_size, total_images):
                    testing_images.append(image_resize[:, :, :])
                    testing_labels.append(cancer_list.index(name))

    training_images = np.asarray(training_images)
    validation_images = np.asarray(validation_images)
    testing_images = np.asarray(testing_images)

    training_labels = np.asarray(training_labels)
    validation_labels = np.asarray(validation_labels)
    testing_labels = np.asarray(testing_labels)

    if encoding == 'Yes':
        labels_count = np.unique(training_labels).shape[0]

        training_labels = dense_to_one_hot(training_labels, labels_count)
        training_labels = training_labels.astype(np.float32)
        validation_labels = dense_to_one_hot(validation_labels, labels_count)
        validation_labels = validation_labels.astype(np.float32)
        testing_labels = dense_to_one_hot(testing_labels, labels_count)
        testing_labels = testing_labels.astype(np.float32)

    print(training_images.shape[0], validation_images.shape[0], testing_images.shape[0])

    return training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels


# 420, 360
num_classes = 4
dropout = 0.35


def xception_model(load_weights=True):
    base_model = Xception(include_top=False, weights='imagenet', input_tensor=None,
                          input_shape=(image_width, image_height, 3), pooling='max')
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.name = 'xception'

    return model


def vgg16_model(load_weights=True):
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=None,
                       input_shape=(image_height, image_width, 3), pooling='max')
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model._name = 'vgg16'

    return model


def vgg19_model(load_weights=True):
    if load_weights:
        base_model = VGG19(include_top=False, weights='imagenet', input_tensor=None,
                           input_shape=(image_height, image_width, 3), pooling='max')  # change from input_shape=(image_width, image_height,3)
    else:
        base_model = VGG19(include_top=False, weights=None, input_tensor=None,
                           input_shape=(image_height, image_width, 3), pooling='max')  # change from input_shape=(image_width, image_height,3)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)   # change from Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model._name = 'vgg19'
    return model


def resnet50_model(load_weights=True):
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                          input_shape=(image_height, image_width, 3), pooling='avg')  # change from input_shape=(image_width, image_height,3)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)   # change from Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.name = 'resnet'
    return model


def inception_model(load_weights=True):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                             input_shape=(image_height, image_width, 3), pooling='avg') # change from input_shape=(image_width, image_height,3)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)   # change from Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.name = 'inception'
    return model


def inception_resnet_model(load_weights=True):
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                   input_shape=(image_height, image_width, 3), pooling='avg') # change from input_shape=(image_width, image_height,3)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)     # change from x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.name = 'inception_resnet'
    return modela


models = [vgg16_model, vgg19_model, xception_model, resnet50_model, inception_model, inception_resnet_model]


def compile_n_fit(validation_percent, testing_percent, load_wt, image_width=175, image_height=115, dropout=0.3,
                  model_name='vgg16_model', magnification='40X'):
    training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels = data_split(
        magnification=magnification, validation_percent=validation_percent, testing_percent=testing_percent)
    for i in range(len(models)):
        if models[i].__name__ == model_name:
            model = models[i]

    model = model(load_weights=load_wt)
    try:
        model.load_weights(model_name + '_weight_1.h5')
        print('Weights loaded!')
    except:
        print('No weights defined!')
    #         pass

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=10, verbose=2)
    model_checkpoint = ModelCheckpoint(model_name + "_combine" + ".model", save_best_only=True, verbose=2)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=2)  # min_lr=0.00001,

    epochs = 100
    batch_size = 32

    history = model.fit(training_images, training_labels,
                        validation_data=(validation_images, validation_labels),
                        epochs=epochs,
                        verbose=0,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr])

    test_loss, test_acc = model.evaluate(testing_images, testing_labels)

    model.save_weights(model_name + '_weight_1.h5')

    print("\nThe test accuracy for " + model_name + " with magnification " + magnification + " is ", test_acc, "\n")


model_num = 1
name = models[model_num].__name__
print("name: " + name)
iteration = 0
for types in magnification_list:
    image_width, image_height = 175, 115
    if iteration == 0:
        load_wt = "Yes"
    else:
        load_wt = "No"
    compile_n_fit(validation_percent=0.15, testing_percent=0.15,
                  image_width=175, image_height=115, dropout=0.3,
                  load_wt=load_wt, model_name=name, magnification=types)
    iteration += 1
