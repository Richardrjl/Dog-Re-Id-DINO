"""
DogFaceNet (MODIFICADO PARA OPTIMIZACIÓN)
Funciones para entrenamiento online.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from math import isnan

SIZE = (224, 224, 3)

# --- FUNCIÓN DE CARGA DE IMÁGENES CORREGIDA ---
def load_images(filenames_batch):
    """
    Carga un LOTE de imágenes usando tf.io y tf.map_fn para mayor velocidad.
    """
    return tf.map_fn(
        lambda filename: tf.image.resize(
            tf.image.convert_image_dtype(
                tf.image.decode_jpeg(tf.io.read_file(filename), channels=3),
                tf.float32
            ),
            SIZE[:2]
        ),
        filenames_batch,
        dtype=tf.float32 # Usar dtype en lugar de fn_output_signature
    )

# --- GENERADOR DE PREDICCIÓN (CORREGIDO) ---
def predict_generator(filenames, batch_size=32):
    """
    Generador para model.predict()
    """
    for i in range(0, len(filenames), batch_size):
        images_batch = load_images(filenames[i:i+batch_size])
        yield (images_batch,)

# --- GENERADOR DE VALIDACIÓN (SOFT TRIPLETS) ---
# (Lo mantenemos para la validación si se desea, aunque no es ideal)
def define_triplets_batch(filenames, labels, nbof_triplet=21 * 3):
    """
    Genera triplets "soft" (aleatorios) offline.
    """
    triplet_train = []
    y_triplet = np.empty(nbof_triplet)
    classes = np.unique(labels)
    for i in range(0, nbof_triplet, 3):
        # Pick a class and chose two pictures from this class
        classAP = classes[np.random.randint(len(classes))]
        keep = np.equal(labels, classAP)
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
        while classN == classAP:
            classN = classes[np.random.randint(len(classes))]
        keep = np.equal(labels, classN)
        keep_classN = filenames[keep]
        keep_classN_idx = labels[keep]
        idx_image3 = np.random.randint(len(keep_classN))
        triplet_train += [keep_classN[idx_image3]]
        y_triplet[i+2] = keep_classN_idx[idx_image3]

    return np.array(triplet_train), y_triplet

def image_generator(filenames, labels, batch_size=63, use_aug=True):
    """
    Generador de entrenamiento para triplets "soft".
    use_aug ahora es ignorado (se hace en el modelo).
    """
    while True:
        f_triplet, y_triplet = define_triplets_batch(filenames, labels, batch_size)
        i_triplet = load_images(f_triplet)
        yield (i_triplet, y_triplet)


# --- GENERADOR DE ENTRENAMIENTO (HARD TRIPLETS) ---
# (Este es el que usará nuestro Callback)

def define_adaptive_hard_triplets_batch(filenames, labels, predict, nbof_triplet=21*3, use_neg=True, use_pos=True):
    """
    Genera triplets "hard" offline.
    """
    assert nbof_triplet % 3 == 0

    _, idx_classes = np.unique(labels, return_index=True)
    classes = labels[np.sort(idx_classes)]

    triplets = []
    y_triplets = np.empty(nbof_triplet)
    
    for i in range(0, nbof_triplet, 3):
        # Elige una clase ancla aleatoria
        class_idx = np.random.randint(len(classes))
        keep = np.equal(labels, classes[class_idx])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Elige una imagen ancla aleatoria de esa clase
        idx_image1 = np.random.randint(len(keep_labels))
        
        # Calcula distancias POSITIVAS
        if use_pos:
            dist_class = np.sum(np.square(predict[keep] - predict[keep][idx_image1]), axis=-1)
            idx_image2 = np.argmax(dist_class) # Hard positive
        else:
            idx_image2 = np.random.randint(len(keep_labels))
            j = 0
            while idx_image1 == idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                j += 1
                if j == 1000: break
        
        triplets += [keep_filenames[idx_image1]]
        y_triplets[i] = keep_labels[idx_image1]
        triplets += [keep_filenames[idx_image2]]
        y_triplets[i+1] = keep_labels[idx_image2]
        
        # Calcula distancias NEGATIVAS
        not_keep = np.logical_not(keep)
        
        if use_neg:
            dist_other = np.sum(np.square(predict[not_keep] - predict[keep][idx_image1]), axis=-1)
            idx_image3 = np.argmin(dist_other) # Hard negative
        else:
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        y_triplets[i+2] = labels[not_keep][idx_image3]

    return np.array(triplets), y_triplets


def online_adaptive_hard_image_generator(
    filenames,                  # Rutas absolutas de las imágenes
    labels,                     # Etiquetas de las imágenes
    predicts,                   # Embeddings PRE-CALCULADOS
    loss,                       # Pérdida actual del modelo
    batch_size      = 63,       # Tamaño de lote (múltiplo de 3)
    use_aug         = True      # Ignorado (aumento en el modelo)
):
    """
    Generador para seleccionar triplets "hard" online adaptativo.
    AHORA USA EMBEDDINGS PRE-CALCULADOS ('predicts')
    """
    hard_triplet_ratio = 0
    nbof_hard_triplets = 0

    while True:
        # Actualizar proporción de hard triplets
        hard_triplet_ratio = np.exp(-loss * 10 / batch_size)
        if np.isnan(hard_triplet_ratio):
            hard_triplet_ratio = 0.5 # Default si la pérdida es NaN
        if hard_triplet_ratio < 0.1:
            hard_triplet_ratio = 0.1 # Mínimo 10% hard
            
        nbof_hard_triplets = int((batch_size // 3) * hard_triplet_ratio)
        nbof_soft_triplets = (batch_size // 3) - nbof_hard_triplets

        # Definir triplets (hard y soft)
        f_triplet_hard, y_triplet_hard = define_adaptive_hard_triplets_batch(
            filenames, labels, predicts,
            nbof_hard_triplets * 3,
            use_neg=True, use_pos=True
        )
        
        f_triplet_soft, y_triplet_soft = define_adaptive_hard_triplets_batch(
            filenames, labels, predicts,
            nbof_soft_triplets * 3,
            use_neg=False, use_pos=False
        )

        f_triplet = np.append(f_triplet_hard, f_triplet_soft)
        y_triplet = np.append(y_triplet_hard, y_triplet_soft)

        # Cargar imágenes de los triplets
        i_triplet = load_images(f_triplet)

        yield (i_triplet, y_triplet)