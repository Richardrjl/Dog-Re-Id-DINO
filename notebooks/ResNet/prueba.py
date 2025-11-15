# dogfacenet_optimized_tfdata.py

import tensorflow as tf
import numpy as np
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DEL PIPELINE DE DATOS
# -----------------------------------------------------------------------------

# Parámetros globales de las imágenes y el batching
SIZE = (224, 224, 3)
# Parámetros para el muestreo: P clases, K imágenes por clase
CLASSES_PER_BATCH = 10  # P
IMAGES_PER_CLASS = 8    # K
# El tamaño final del lote será P * K
BATCH_SIZE = CLASSES_PER_BATCH * IMAGES_PER_CLASS


@tf.function
def load_and_preprocess_image(path, label):
    """
    Carga una imagen desde su ruta, la decodifica, cambia su tamaño y la normaliza.
    Esta función se ejecuta en el grafo de TensorFlow para máxima eficiencia.
    """
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, [SIZE[0], SIZE[1]])
    image = image / 255.0  # Normalizar a [0, 1]
    return image, label


class PKImageSampler:
    """
    Generador de Python para el muestreo P-K.
    
    Este generador solo define QUÉ archivos cargar; tf.data se encarga de CÓMO
    cargarlos de manera eficiente.
    """
    def __init__(self, filenames, labels, classes_per_batch, images_per_class):
        self.filenames = np.array(filenames)
        self.labels = np.array(labels)
        self.classes_per_batch = classes_per_batch
        self.images_per_class = images_per_class

        # Pre-calcular el mapa de índices por clase para búsquedas rápidas
        self.indices_by_class = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.indices_by_class[label].append(i)
        
        self.unique_classes = list(self.indices_by_class.keys())
        print(f"Muestreador inicializado con {len(self.unique_classes)} clases únicas.")

    def __call__(self):
        """
        El generador que produce las rutas y etiquetas para un lote.
        Se ejecutará indefinidamente.
        """
        while True:
            # 1. Seleccionar P clases al azar sin reemplazo
            selected_classes = np.random.choice(
                self.unique_classes, self.classes_per_batch, replace=False
            )

            # 2. Para cada clase, seleccionar K imágenes al azar
            for cls in selected_classes:
                class_indices = self.indices_by_class[cls]
                # Usar replace=True si una clase puede tener menos de K imágenes
                # Esto asegura que siempre se generen K imágenes por clase.
                selected_indices = np.random.choice(
                    class_indices, self.images_per_class, replace=True
                )
                
                for idx in selected_indices:
                    yield self.filenames[idx], self.labels[idx]


def create_dataset(filenames, labels, classes_per_batch, images_per_class):
    """
    Construye y devuelve un pipeline de tf.data optimizado.
    """
    batch_size = classes_per_batch * images_per_class
    sampler = PKImageSampler(filenames, labels, classes_per_batch, images_per_class)

    # Crear el dataset a partir del generador de Python
    dataset = tf.data.Dataset.from_generator(
        sampler,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),   # Ruta del archivo
            tf.TensorSpec(shape=(), dtype=tf.int32)     # Etiqueta
        )
    )
    
    # Mapear la función de carga y preprocesamiento en paralelo
    # AUTOTUNE ajustará el nivel de paralelismo dinámicamente.
    dataset = dataset.map(
        load_and_preprocess_image, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Agrupar las imágenes en lotes del tamaño P * K
    dataset = dataset.batch(batch_size)
    
    # Poner en cola el siguiente lote mientras la GPU está ocupada (prefetching)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# -----------------------------------------------------------------------------
# 2. FUNCIÓN DE PÉRDIDA PERSONALIZADA (BATCH HARD TRIPLET LOSS)
# -----------------------------------------------------------------------------

class BatchHardTripletLoss(tf.keras.losses.Loss):
    """
    Calcula la pérdida triplet "batch-hard".

    Para cada ancla en el lote, encuentra el positivo más lejano (hard positive)
    y el negativo más cercano (hard negative) para formar el triplet.
    """
    def __init__(self, margin=0.5, name="BatchHardTripletLoss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # y_true son las etiquetas (shape: [batch_size])
        # y_pred son los embeddings (shape: [batch_size, embedding_dim])
        
        labels = tf.reshape(y_true, [-1, 1])
        embeddings = tf.math.l2_normalize(y_pred, axis=1) # Normalizar embeddings es una buena práctica

        # 1. Calcular la matriz de distancias al cuadrado
        pairwise_dist = self._pairwise_distances(embeddings)

        # 2. Crear una máscara para identificar pares positivos
        mask_anchor_positive = tf.equal(labels, tf.transpose(labels))
        mask_anchor_positive = tf.cast(mask_anchor_positive, dtype=tf.float32)

        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # 3. Encontrar la distancia al positivo más lejano (hardest positive)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

        # 4. Encontrar la distancia al negativo más cercano (hardest negative)
        mask_anchor_negative = tf.logical_not(tf.cast(mask_anchor_positive, dtype=tf.bool))
        mask_anchor_negative = tf.cast(mask_anchor_negative, dtype=tf.float32)
        
        # Añadimos un valor grande a los positivos para ignorarlos al buscar el mínimo
        max_dist = tf.reduce_max(pairwise_dist)
        anchor_negative_dist = pairwise_dist + max_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        # 5. Calcular la pérdida triplet
        # loss = max(d(A, P) - d(A, N) + margin, 0)
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)

        return tf.reduce_mean(triplet_loss)

    def _pairwise_distances(self, embeddings):
        """Calcula la matriz de distancias Euclidianas al cuadrado de forma eficiente."""
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
        square_norm = tf.linalg.diag_part(dot_product)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
        return tf.maximum(distances, 0.0)
