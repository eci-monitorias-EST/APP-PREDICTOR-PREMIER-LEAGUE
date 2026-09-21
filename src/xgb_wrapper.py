"""
xgb_wrapper.py

Tu versión de xgboost (a diferencia de RandomForestClassifier y MLPClassifier
de sklearn) NO acepta etiquetas de texto ('A','D','H') directamente en .fit():
exige enteros 0,1,2. Este wrapper resuelve eso de forma transparente:
convierte las etiquetas a enteros antes de entrenar y las devuelve a texto
en predict()/classes_, para que 03 y 04 puedan tratarlo exactamente igual
que a RandomForestClassifier o MLPClassifier (mismo .fit, .predict_proba,
.classes_), sin tener que tocar el resto del pipeline ni app.py.

CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR
--------------------------------------
1) class_weight="balanced"
   XGBoost no tiene el parámetro class_weight de sklearn. El equivalente es
   pasarle un peso POR FILA en .fit(sample_weight=...). Con
   compute_sample_weight("balanced", y) cada clase recibe peso
   n_muestras / (n_clases * n_de_esa_clase), que es exactamente la misma
   fórmula que usa class_weight="balanced" en sklearn. Así el empate deja de
   valer menos que el resto y el modelo puede llegar a predecirlo.

2) X se convierte a numpy float32 antes de entrenar/predecir
   La corrida anterior murió en 3 de 300 fits con:
       OSError: exception: access violation reading 0x0000000000000008
   dentro de XGProxyDMatrixCreate / _ref_data_from_columnar. Ese es el camino
   que usa xgboost cuando recibe un DataFrame de pandas y arma la matriz por
   columnas (QuantileDMatrix). Pasándole un array de numpy se evita ese
   camino por completo.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight


class XGBClassifierWithStringLabels(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 subsample=1.0, colsample_bytree=1.0, random_state=None,
                 class_weight=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.class_weight = class_weight

    @staticmethod
    def _as_array(X):
        # np.asarray sobre un DataFrame devuelve el bloque de numpy subyacente.
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier

        X = self._as_array(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)  # orden alfabético: ['A','D','H']
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([label_to_idx[v] for v in y])

        # class_weight="balanced" -> peso por fila, el equivalente en xgboost
        if sample_weight is None and self.class_weight == "balanced":
            sample_weight = compute_sample_weight("balanced", y_encoded)

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=len(self.classes_),
            eval_metric="mlogloss",
            random_state=self.random_state,
        )
        self.model_.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._as_array(X))

    def predict(self, X):
        idx = self.model_.predict(self._as_array(X))
        return self.classes_[idx]
