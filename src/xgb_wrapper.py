"""
xgb_wrapper.py
Tu versión de xgboost (a diferencia de RandomForestClassifier y MLPClassifier
de sklearn) NO acepta etiquetas de texto ('A','D','H') directamente en .fit():
exige enteros 0,1,2. Este wrapper resuelve eso de forma transparente:
convierte las etiquetas a enteros antes de entrenar y las devuelve a texto
en predict()/classes_, para que 03 y 04 puedan tratarlo exactamente igual
que a RandomForestClassifier o MLPClassifier (mismo .fit, .predict_proba,
.classes_), sin tener que tocar el resto del pipeline ni app.py.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBClassifierWithStringLabels(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 subsample=1.0, colsample_bytree=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier
        y = np.asarray(y)
        self.classes_ = np.unique(y)  # orden alfabético: ['A','D','H']
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([label_to_idx[v] for v in y])

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
        self.model_.fit(X, y_encoded)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        idx = self.model_.predict(X)
        return self.classes_[idx]
