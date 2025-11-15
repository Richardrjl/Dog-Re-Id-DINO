"""
DogFaceNet
Functions for training on bigger datasets then offline_training module.
It does not load all the dataset into memory but just a part of it.
It mainly relies on keras data generators.
It contains:
 - Offline triplet generator: for soft and hard triplets
 - Online triplet generator: for soft and hard triplets

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import pickle
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential
from math import isnan

SIZE = (224,224,3)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=8,
    zoom_range=0.1,
    fill_mode='nearest',
    channel_shift_range = 0.1
)

def apply_transform(images, datagen):
    """
    Aplica una tranformación de procesamiento de datos a n imagenes
    Args:
        -Imagenes
        -ImageDataGenerator
    Return:
        -Imagenes del mismo tamaño de entrada pero transformadas
    """
    for x in datagen.flow(images, batch_size=len(images), shuffle=False):
        return x

def define_triplets_batch(filenames,labels,nbof_triplet = 21 * 3):
    """
    Generates offline soft triplet.
    Given a list of file names of pictures, their specific label and
    a number of triplet images, returns an array of triplet of images
    and their specific labels.

    Args:
     - filenames: array of strings. List of file names of the pictures. 
     - labels: array of integers.
     - nbof_triplet: integer. Has to be a multiple of 3.
     
     Returns:
     - triplet_train: array of pictures --> a 4D array. 
     - y_triplet: array of integers of same dimension as the first
     dimension of triplet_train. Contains the labels of the pictures.
    """
    triplet_train = []
    y_triplet = np.empty(nbof_triplet)
    classes = np.unique(labels)
    for i in range(0,nbof_triplet,3):
        # Pick a class and chose two pictures from this class
        classAP = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classAP)
        keep_classAP = filenames[keep]
        keep_classAP_idx = labels[keep]
        idx_image1 = np.random.randint(len(keep_classAP))
        idx_image2 = np.random.randint(len(keep_classAP))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(keep_classAP))

        triplet_train += [keep_classAP[idx_image1]]
        triplet_train += [keep_classAP[idx_image2]]
        y_triplet[i] = keep_classAP_idx[idx_image1]
        y_triplet[i+1] = keep_classAP_idx[idx_image2]
        # Pick a class for the negative picture
        classN = classes[np.random.randint(len(classes))]
        while classN==classAP:
            classN = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classN)
        keep_classN = filenames[keep]
        keep_classN_idx = labels[keep]
        idx_image3 = np.random.randint(len(keep_classN))
        triplet_train += [keep_classN[idx_image3]]
        y_triplet[i+2] = keep_classN_idx[idx_image3]
        
    return triplet_train, y_triplet

# def define_hard_triplets_batch(filenames,labels,predict,nbof_triplet=21*3, use_neg=True, use_pos=True):
#     """
#     [DEPRECATED] Use define_adaptive_hard_triplets_batch instead!
#     Generates hard triplet for offline selection. It will consider the whole dataset.
    
#     Args:
#         -images: images from which the triplets will be created
#         -labels: labels of the images
#         -predict: predicted embeddings for the images by the trained model
#         -alpha: threshold of the triplet loss
#     Returns:
#         -triplet
#         -y_triplet: labels of the triplets
#     """
#     # Check if we have the right number of triplets
#     assert nbof_triplet%3 == 0
    
#     _,idx_classes = np.unique(labels,return_index=True)
#     classes = labels[np.sort(idx_classes)]
    
#     triplets = []
#     y_triplets = np.empty(nbof_triplet)
    
#     for i in range(0,nbof_triplet,3):
#         # Chooses the first class randomly
#         keep = np.equal(labels,classes[np.random.randint(len(classes))])
#         keep_filenames = filenames[keep]
#         keep_labels = labels[keep]
        
#         # Chooses the first image among this class randomly
#         idx_image1 = np.random.randint(len(keep_labels))
        
        
#         # Computes the distance between the chosen image and the rest of the class
#         if use_pos:
#             dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)

#             idx_image2 = np.argmax(dist_class)
#         else:
#             idx_image2 = np.random.randint(len(keep_labels))
#             i = 0
#             while idx_image1==idx_image2:
#                 idx_image2 = np.random.randint(len(keep_labels))
#                 # Just to prevent endless loop:
#                 i += 1
#                 if i == 1000:
#                     print("[Error: define_hard_triplets_batch] Endless loop.")
#                     break
        
#         triplets += [keep_filenames[idx_image1]]
#         y_triplets[i] = keep_labels[idx_image1]
#         triplets += [keep_filenames[idx_image2]]
#         y_triplets[i+1] = keep_labels[idx_image2]
        
        
#         # Computes the distance between the chosen image and the rest of the other classes
#         not_keep = np.logical_not(keep)
        
#         if use_neg:
#             dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
#             idx_image3 = np.argmin(dist_other) 
#         else:
#             idx_image3 = np.random.randint(len(filenames[not_keep]))
            
#         triplets += [filenames[not_keep][idx_image3]]
#         y_triplets[i+2] = labels[not_keep][idx_image3]

#     #return triplets, y_triplets
#     return np.array(triplets), y_triplets

def define_adaptive_hard_triplets_batch(filenames,labels,predict,nbof_triplet=21*3, use_neg=True, use_pos=True):
    """
    Generates hard triplet for offline selection. It will consider the whole dataset.
    This function will also return the predicted values.
    
    Args:
        -images: images from which the triplets will be created
        -labels: labels of the images
        -predict: predicted embeddings for the images by the trained model
        -alpha: threshold of the triplet loss
    Returns:
        -triplets
        -y_triplets: labels of the triplets
        -pred_triplets: predicted embeddings of the triplets
    """
    # Check if we have the right number of triplets
    assert nbof_triplet%3 == 0
    
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    
    triplets = []
    y_triplets = np.empty(nbof_triplet)
    pred_triplets = np.empty((nbof_triplet,predict.shape[-1]))
    
    for i in range(0,nbof_triplet,3):
        # Chooses the first class randomly
        keep = np.equal(labels,classes[np.random.randint(len(classes))])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Chooses the first image among this class randomly
        idx_image1 = np.random.randint(len(keep_labels))
        
        
        # Computes the distance between the chosen image and the rest of the class
        if use_pos:
            dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)

            idx_image2 = np.argmax(dist_class)
        else:
            idx_image2 = np.random.randint(len(keep_labels))
            j = 0
            while idx_image1==idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                # Just to prevent endless loop:
                j += 1
                if j == 1000:
                    print("[Error: define_hard_triplets_batch] Endless loop.")
                    break
        
        triplets += [keep_filenames[idx_image1]]
        y_triplets[i] = keep_labels[idx_image1]
        pred_triplets[i] = predict[keep][idx_image1]
        triplets += [keep_filenames[idx_image2]]
        y_triplets[i+1] = keep_labels[idx_image2]
        pred_triplets[i+1] = predict[keep][idx_image2]
        
        # Computes the distance between the chosen image and the rest of the other classes
        not_keep = np.logical_not(keep)
        
        if use_neg:
            dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
            idx_image3 = np.argmin(dist_other) 
        else:
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        y_triplets[i+2] = labels[not_keep][idx_image3]
        pred_triplets[i+2] = predict[not_keep][idx_image3]

    return np.array(triplets), y_triplets, pred_triplets

def load_images(filenames_batch):
    """
    Carga un LOTE de imágenes usando tf.io y tf.map_fn para mayor velocidad.
    """
    # tf.map_fn aplica la función anónima a cada elemento en filenames_batch
    return tf.map_fn(
        lambda filename: tf.image.resize(
            tf.image.convert_image_dtype(
                tf.image.decode_jpeg(tf.io.read_file(filename), channels=3),
                tf.float32
            ),
            SIZE[:2]
        ),
        filenames_batch,
        fn_output_signature=tf.float32
    )

def create_dataset(filenames, labels, batch_size=63, use_aug=True):
    # Crear un dataset con las rutas y etiquetas
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Barajar los datos para asegurar aleatoriedad en los lotes
    dataset = dataset.shuffle(buffer_size=1024)

    # Aquí iría la lógica para formar los tripletes.
    # Esta es la parte más compleja de adaptar, pero se puede hacer
    # con .group_by_window(), .map(), y .filter().
    # Por simplicidad, este ejemplo solo muestra carga de imágenes.
    
    # Mapear la función de carga y procesamiento en paralelo
    dataset = dataset.map(lambda x, y: (load_images(x), y),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Si se usa aumento de datos
    if use_aug:
        # Es mejor usar capas de preprocesamiento de Keras directamente en el modelo
        # o funciones de tf.image aquí para mantener todo en el grafo.
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Crear lotes de datos
    dataset = dataset.batch(batch_size)

    # Activar el prefetching para un rendimiento óptimo
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def hard_image_generator(filenames, labels, predict, batch_size=63, use_neg=True, use_pos=True, use_aug=True, datagen=datagen):
    """
    Training generator for offline hard triplets.
    """
    while True:
        f_triplet, y_triplet = define_adaptive_hard_triplets_batch(filenames, labels, predict, batch_size, use_neg=use_neg, use_pos=use_pos)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def predict_generator(filenames, batch_size=32):
    """
    Prediction generator.
    """
    for i in range(0,len(filenames),batch_size):
        images_batch = load_images(filenames[i:i+batch_size])
        yield (images_batch,)

def online_hard_image_generator(
    filenames,
    labels,
    model,
    batch_size=63,
    nbof_subclasses=10,
    use_neg=True,
    use_pos=True,
    use_aug=True,
    datagen=datagen):
    """
    Generator to select online hard triplets for training.
    
    Arguments:
        -filenames
        -labels
    """
    while True:
        # Select a certain amount of subclasses
        classes = np.unique(labels)
        subclasses = np.random.choice(classes,size=nbof_subclasses,replace=False)
        
        keep_classes = np.equal(labels,subclasses[0])
        for i in range(1,len(subclasses)):
            keep_classes = np.logical_or(keep_classes,np.equal(labels,subclasses[i]))
        subfilenames = filenames[keep_classes]
        sublabels = labels[keep_classes]
        predict = model.predict(predict_generator(subfilenames, 32),
                                          steps=np.ceil(len(subfilenames)/32))
        
        f_triplet, y_triplet = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, batch_size, use_neg=use_neg, use_pos=use_pos)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def online_adaptive_hard_image_generator(
    filenames,                  # Absolute path of the images
    labels,                     # Labels of the images
    model,                      # A keras model
    loss,                       # Current loss of the model
    batch_size      = 63,       # Batch size (has to be a multiple of 3 for dogfacenet)
    nbof_subclasses = 10,       # Number of subclasses from which the triplets will be selected
    use_aug         = True,     # Use data augmentation?
    datagen         = datagen   # Data augmentation parameter
):
    """
    Generator to select online hard triplets for training.
    Include an adaptive control on the number of hard triplets included during the training.
    """

    hard_triplet_ratio = 0
    nbof_hard_triplets = 0

    while True:
        # Seleccionar un subconjunto de clases
        classes = np.unique(labels)
        subclasses = np.random.choice(
            classes,
            size=int(nbof_subclasses * hard_triplet_ratio) + 2,
            replace=False
        )

        # Filtrar imágenes según esas clases
        keep_classes = np.equal(labels, subclasses[0])
        for i in range(1, len(subclasses)):
            keep_classes = np.logical_or(keep_classes, np.equal(labels, subclasses[i]))
        subfilenames = filenames[keep_classes]
        sublabels = labels[keep_classes]

        predict = model.predict(
            predict_generator(subfilenames, 32),
            steps=int(np.ceil(len(subfilenames) / 32)),
            verbose=0
        )

        # Definir triplets (hard y soft)
        f_triplet_hard, y_triplet_hard, predict_hard = define_adaptive_hard_triplets_batch(
            subfilenames, sublabels, predict,
            nbof_hard_triplets * 3,
            use_neg=True, use_pos=True
        )
        f_triplet_soft, y_triplet_soft, predict_soft = define_adaptive_hard_triplets_batch(
            subfilenames, sublabels, predict,
            batch_size - nbof_hard_triplets * 3,
            use_neg=False, use_pos=False
        )

        f_triplet = np.append(f_triplet_hard, f_triplet_soft)
        y_triplet = np.append(y_triplet_hard, y_triplet_soft)
        predict = np.append(predict_hard, predict_soft, axis=0)

        # Actualizar proporción de hard triplets
        hard_triplet_ratio = np.exp(-loss * 10 / batch_size)
        if np.isnan(hard_triplet_ratio):
            hard_triplet_ratio = 0
        nbof_hard_triplets = int(batch_size // 3 * hard_triplet_ratio)

        # Cargar imágenes de los triplets
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)

        yield (i_triplet, y_triplet)