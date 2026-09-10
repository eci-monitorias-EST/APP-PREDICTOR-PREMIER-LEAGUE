"""
keras_model_builder.py
Define la arquitectura de la red neuronal en Keras/TensorFlow (en vez del
MLPClassifier de scikit-learn, que no soporta dropout). Lo usan tanto
03_train_compare_models.py (para la búsqueda de hiperparámetros) como
04_retrain_final_model.py (para reconstruir el modelo ganador).
"""
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers


def build_keras_model(hidden_layer_sizes=(16,), activation="relu",
                       alpha=0.0001, dropout=0.0, meta=None):
    """
    hidden_layer_sizes: tupla con el número de neuronas por capa oculta,
                         igual que en MLPClassifier. Ej: (32, 16) = 2 capas.
    activation:          'relu' o 'linear' (identidad) en las capas ocultas.
    alpha:               fuerza de la regularización L2 (penaliza pesos grandes).
    dropout:             proporción de neuronas que se "apagan" al azar en
                          cada paso de entrenamiento (0.0 = desactivado).
    meta:                scikeras inyecta aquí automáticamente info del
                          dataset (n_features_in_, n_classes_) al momento
                          de construir el modelo - no hay que pasarlo a mano.
    """
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]

    model = keras.Sequential()
    model.add(keras.Input(shape=(n_features_in_,)))
    for units in hidden_layer_sizes:
        model.add(layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizers.l2(alpha),
        ))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes_, activation="softmax"))

    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01),  # descenso de gradiente clásico
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
