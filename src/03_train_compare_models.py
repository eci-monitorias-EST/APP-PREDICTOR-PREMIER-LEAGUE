"""
03_train_compare_models.py

Etapa A del plan acordado:
  - Train: temporadas hasta 2024-2025 (inclusive)
  - Test:  temporada 2025-2026 (backtest histórico, ya jugada por completo)

Comparamos 3 familias de modelos:
  - Random Forest      (ya lo conocías)
  - XGBoost             (nuevo)  <- instala con: pip install xgboost
  - Red neuronal (MLP)  (nuevo)

Métricas: ROC-AUC macro (OVR) como principal -por el desbalance H/D/A-,
más log-loss (calibración de probabilidades) y accuracy/classification_report
de referencia.

Si xgboost no está instalado, este script cae automáticamente a
HistGradientBoostingClassifier (sklearn) como proxy, para que el pipeline
nunca se rompa. En tu máquina, instala xgboost ANTES de correr esto para
tener la comparación real.

Ejecutar desde cualquier lado con: python src/03_train_compare_models.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  # ya no se usa para la red final,
                                                    # se deja por si quieres comparar
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from xgboost import XGBClassifier
    from xgb_wrapper import XGBClassifierWithStringLabels
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[AVISO] xgboost no está instalado -> `pip install xgboost` para la comparación real.")
    print("        Mientras tanto se usa HistGradientBoostingClassifier (sklearn) como proxy.")
    from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 2658

# =========================================
# 1. CARGA Y SPLIT TEMPORAL
# =========================================
df = pd.read_csv(PROCESSED_DIR / "dfpremier_features.csv")

FEATURES = ["elo_diff", "winrate_diff"]
LABEL = "result"

train_df = df[df["season"] <= "2024-2025"].copy()
test_df = df[df["season"] == "2025-2026"].copy()

print(f"Train: {train_df.shape[0]} partidos (hasta 2024-2025)")
print(f"Test:  {test_df.shape[0]} partidos (2025-2026)")

X_train, y_train = train_df[FEATURES], train_df[LABEL]
X_test, y_test = test_df[FEATURES], test_df[LABEL]

CLASSES = sorted(y_train.unique())  # ['A', 'D', 'H']
print("Clases:", CLASSES)

tscv = TimeSeriesSplit(n_splits=3)


def evaluate(model, X, y, name, classes=CLASSES):
    proba = model.predict_proba(X)
    pred = model.predict(X)
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    ll = log_loss(y, proba, labels=classes)
    acc = accuracy_score(y, pred)
    print(f"\n----- {name} -----")
    print(f"ROC-AUC (macro, OVR): {auc:.4f}")
    print(f"Log-loss:             {ll:.4f}")
    print(f"Accuracy:             {acc:.4f}")
    print(classification_report(y, pred, labels=classes))
    return {"model": name, "roc_auc_macro": auc, "log_loss": ll, "accuracy": acc}


results = []

# =========================================
# 2. RANDOM FOREST
# =========================================
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"],
}

rf_grid = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=rf_param_grid,
    n_iter=20,
    cv=tscv,
    n_jobs=-1,
    scoring="roc_auc_ovr",
    refit=True,
    random_state=RANDOM_STATE,
)
rf_grid.fit(X_train, y_train)
print("Mejores hiperparámetros RF:", rf_grid.best_params_)
best_rf = rf_grid.best_estimator_
results.append(evaluate(best_rf, X_test, y_test, "Random Forest (test 2025-2026)"))

# =========================================
# 3. XGBOOST (o su proxy HistGradientBoosting si no está instalado)
# =========================================
if HAS_XGB:
    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)

    xgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    xgb_grid = RandomizedSearchCV(
        estimator=XGBClassifierWithStringLabels(random_state=RANDOM_STATE),
        param_distributions=xgb_param_grid,
        n_iter=20,
        cv=tscv,
        n_jobs=-1,
        scoring="roc_auc_ovr",
        refit=True,
        random_state=RANDOM_STATE,
    )
    # El wrapper acepta y con etiquetas string ('A','D','H') directamente.
    xgb_grid.fit(X_train, y_train)
    print("Mejores hiperparámetros XGB:", xgb_grid.best_params_)
    best_xgb = xgb_grid.best_estimator_
    assert list(best_xgb.classes_) == CLASSES, "Orden de clases inesperado en XGBoost"
    results.append(evaluate(best_xgb, X_test, y_test, "XGBoost (test 2025-2026)"))
else:
    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING (proxy de XGBoost - HistGradientBoostingClassifier)")
    print("=" * 60)

    gb_param_grid = {
        "max_iter": [100, 200, 300],
        "max_depth": [2, 3, 4, None],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_regularization": [0.0, 0.1, 1.0],
    }

    xgb_grid = RandomizedSearchCV(
        estimator=HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=gb_param_grid,
        n_iter=20,
        cv=tscv,
        n_jobs=-1,
        scoring="roc_auc_ovr",
        refit=True,
        random_state=RANDOM_STATE,
    )
    xgb_grid.fit(X_train, y_train)
    print("Mejores hiperparámetros (proxy GB):", xgb_grid.best_params_)
    best_xgb = xgb_grid.best_estimator_
    results.append(evaluate(best_xgb, X_test, y_test, "Gradient Boosting proxy (test 2025-2026)"))

# =========================================
# 4. RED NEURONAL (Keras / TensorFlow, con dropout real)
#    OJO: cambiamos de MLPClassifier (sklearn) a Keras porque sklearn NO
#    soporta dropout. Esto agrega tensorflow y scikeras como dependencias
#    nuevas (pip install -r requirements.txt ya las incluye).
# =========================================
print("\n" + "=" * 60)
print("RED NEURONAL (Keras / TensorFlow)")
print("=" * 60)

from scikeras.wrappers import KerasClassifier
from keras_model_builder import build_keras_model

keras_clf = KerasClassifier(
    model=build_keras_model,
    hidden_layer_sizes=(16,),
    activation="relu",
    alpha=0.0001,
    dropout=0.0,
    batch_size=32,
    epochs=100,
    verbose=0,
    random_state=RANDOM_STATE,
)

keras_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", keras_clf),
])

keras_param_grid = {
    "clf__hidden_layer_sizes": [(16,), (32, 16), (64,), (64, 32)],
    "clf__activation": ["relu", "linear"],
    "clf__alpha": [0.0001, 0.001, 0.01],
    "clf__dropout": [0.0, 0.3],
    # batch_size >= al tamaño de train equivale a "batch completo"
    # (descenso de gradiente clásico, no estocástico); 32/128/256 son mini-batch.
    "clf__batch_size": [32, 128, 256, 8192],
}

keras_grid = RandomizedSearchCV(
    estimator=keras_pipeline,
    param_distributions=keras_param_grid,
    n_iter=12,
    cv=tscv,
    n_jobs=1,  # TensorFlow no es seguro para paralelizar con n_jobs=-1
    scoring="roc_auc_ovr",
    refit=True,
    random_state=RANDOM_STATE,
)
keras_grid.fit(X_train, y_train)

# Guardamos los hiperparámetros ganadores con nombres "limpios" (sin el
# prefijo clf__ de la Pipeline) para poder reutilizarlos tal cual
# en 04_retrain_final_model.py
best_keras_params = {k.split("__")[-1]: v for k, v in keras_grid.best_params_.items()}
print("Mejores hiperparámetros Red Neuronal (Keras):", best_keras_params)

best_keras = keras_grid.best_estimator_  # Pipeline completo (scaler + clf)
results.append(evaluate(best_keras, X_test, y_test, "Red Neuronal Keras (test 2025-2026)"))

# =========================================
# 5. TABLA COMPARATIVA FINAL
# =========================================
print("\n" + "=" * 60)
print("COMPARATIVA FINAL (test = temporada 2025-2026)")
print("=" * 60)
results_df = pd.DataFrame(results).sort_values("roc_auc_macro", ascending=False)
print(results_df.to_string(index=False))

results_df.to_csv(REPORTS_DIR / "model_comparison_results.csv", index=False)

best_params = {"rf": rf_grid.best_params_, "mlp_keras": best_keras_params}
best_params["xgb" if HAS_XGB else "gb_proxy"] = xgb_grid.best_params_

with open(REPORTS_DIR / "best_hyperparams.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)

print(f"\nGuardado en {REPORTS_DIR}/model_comparison_results.csv y best_hyperparams.json")
print("\n>>> Siguiente paso: python src/04_retrain_final_model.py")