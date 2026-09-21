"""
03_train_compare_models.py

Compara Random Forest, XGBoost y una red neuronal (Keras) sobre el mismo
split temporal:
  TRAIN: temporadas <= 2024-2025
  TEST : temporada 2025-2026 (backtest, ya jugada por completo)

POR QUÉ EL SCORING ES neg_log_loss Y NO roc_auc_ovr
---------------------------------------------------
La app no predice un resultado: muestra P(local), P(empate), P(visitante).
El entregable son las PROBABILIDADES, así que hay que optimizar la calidad de
la probabilidad, no la del acierto.

  - log-loss es una regla de puntuación propia: su mínimo está exactamente en
    la probabilidad verdadera. Si el modelo dice 27% de empate y empatan el
    27% de las veces, el log-loss lo premia. Si dice 33%, lo castiga.
  - roc_auc_ovr solo mide el ORDEN de las probabilidades, no su valor. Un
    modelo que multiplique todas las P(empate) por 1.5 tiene el mismo AUC y
    una app peor.

Consecuencia práctica: con neg_log_loss, class_weight="balanced" queda en la
grilla pero el propio GridSearch lo descarta, porque reponderar las clases
infla P(empate) por encima de la tasa real. Medido sobre estos datos:

    class_weight=None      -> P(D) media 0.246 | log-loss 1.0389
    class_weight=balanced  -> P(D) media 0.335 | log-loss 1.0550
    tasa real de empates   -> 0.274

SOBRE "EL MODELO NUNCA PREDICE EMPATES"
---------------------------------------
Aparece en el classification_report y NO es un defecto a corregir. Para que
argmax elija D hace falta P(D) > 1/3: si P(D) <= 1/3 entonces P(A)+P(H) >= 2/3
y una de las dos le gana siempre. Como el empate ocurre el ~27% de las veces,
un modelo bien calibrado casi nunca pasa de 1/3. El colapso es consecuencia de
estar bien calibrado, no de estar mal entrenado, y la app no se ve afectada
porque nunca llama a .predict(). Por eso este script reporta n_pred_D (para
dejarlo documentado) y una tabla de calibración (que es la evaluación que sí
corresponde al entregable).

GUARDADO INCREMENTAL
--------------------
Antes el CSV se escribía en la última línea: si Keras fallaba se perdían
también RF y XGBoost. Ahora se guarda después de CADA modelo.

Ejecutar desde cualquier lado con: python src/03_train_compare_models.py
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, accuracy_score,
    balanced_accuracy_score, f1_score, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xgb_wrapper import XGBClassifierWithStringLabels  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 2658
FEATURES = ["elo_diff", "winrate_diff"]
LABEL = "result"
TEST_SEASON = "2025-2026"

# =========================================
# 1. CARGA Y SPLIT TEMPORAL
# =========================================
df = pd.read_csv(PROCESSED_DIR / "dfpremier_features.csv")
train_df = df[df["season"] < TEST_SEASON].copy()
test_df = df[df["season"] == TEST_SEASON].copy()

X_train, y_train = train_df[FEATURES], train_df[LABEL]
X_test, y_test = test_df[FEATURES], test_df[LABEL]
CLASSES = sorted(y_train.unique())          # ['A', 'D', 'H']
IDX_D = CLASSES.index("D")

print(f"Train: {len(train_df)} partidos (hasta {TEST_SEASON} exclusive)")
print(f"Test : {len(test_df)} partidos ({TEST_SEASON})")
print("Reparto train:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Reparto test :", y_test.value_counts(normalize=True).round(3).to_dict())

tscv = TimeSeriesSplit(n_splits=3)
TASA_D_TEST = float((y_test == "D").mean())

# Referencias contra las que hay que compararse. Un modelo que no le gane a
# esto no está aportando nada.
p_base = np.array([[(y_train == c).mean() for c in CLASSES]] * len(y_test))
BASE_LOGLOSS = log_loss(y_test, p_base, labels=CLASSES)
BASE_ACC_H = float((y_test == "H").mean())
print(f"\nBaselines en test -> siempre 'gana local': accuracy {BASE_ACC_H:.4f} | "
      f"predecir la tasa base: log-loss {BASE_LOGLOSS:.4f}")


# =========================================
# 2. EVALUACIÓN
# =========================================
def evaluate(model, X, y, name):
    proba = model.predict_proba(X)
    pred = model.predict(X)
    pD = proba[:, IDX_D]

    row = {
        "model": name,
        "log_loss": log_loss(y, proba, labels=CLASSES),
        "roc_auc_macro": roc_auc_score(y, proba, multi_class="ovr",
                                       average="macro", labels=CLASSES),
        "brier_D": brier_score_loss((np.asarray(y) == "D").astype(int), pD),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
        "P_D_media": float(pD.mean()),
        "P_D_max": float(pD.max()),
        "sesgo_P_D": float(pD.mean() - TASA_D_TEST),   # + = infla el empate
        "n_pred_D": int((pred == "D").sum()),
    }

    print(f"\n----- {name} -----")
    print(f"Log-loss:          {row['log_loss']:.4f}   (baseline {BASE_LOGLOSS:.4f})")
    print(f"ROC-AUC (macro):   {row['roc_auc_macro']:.4f}")
    print(f"Brier del empate:  {row['brier_D']:.4f}")
    print(f"Accuracy:          {row['accuracy']:.4f}   (siempre local {BASE_ACC_H:.4f})")
    print(f"P(D) media {row['P_D_media']:.4f} vs tasa real {TASA_D_TEST:.4f} "
          f"-> sesgo {row['sesgo_P_D']:+.4f}")
    print(f"P(D) máxima observada: {row['P_D_max']:.4f} "
          f"(argmax necesita > 0.3333 para elegir empate)")
    print(f"Empates predichos por argmax: {row['n_pred_D']}")
    print(classification_report(y, pred, labels=CLASSES, zero_division=0))
    return row


def tabla_calibracion(proba, y, nombre, n_bins=4):
    """Parte los partidos por P(D) y compara contra la tasa observada.

    Esta es la evaluación que corresponde al entregable: si la app dice 30%,
    ¿empatan de verdad el 30% de esos partidos? Ojo con el tamaño de muestra:
    con 380 partidos y 4 grupos son ~95 por grupo, y el error estándar de una
    proporción cerca de 0.27 con n=95 es ~0.046. Diferencias menores a ~9
    puntos entre grupos son ruido, no descalibración.
    """
    pD = proba[:, IDX_D]
    d = pd.DataFrame({"P_D": pD, "empate": (np.asarray(y) == "D").astype(int)})
    d["grupo"] = pd.qcut(d["P_D"], n_bins, duplicates="drop")
    g = d.groupby("grupo", observed=True).agg(
        n=("empate", "size"), P_D_media=("P_D", "mean"), tasa_real=("empate", "mean"))
    g["error_estandar"] = np.sqrt(g["tasa_real"] * (1 - g["tasa_real"]) / g["n"])
    print(f"\n  --- Calibración de P(empate) — {nombre} ---")
    print(g.round(3).to_string())
    return g


results, best_params, proba_cols = [], {}, {}


def save_progress():
    pd.DataFrame(results).sort_values("log_loss").to_csv(
        REPORTS_DIR / "model_comparison_results.csv", index=False)
    with open(REPORTS_DIR / "best_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    if proba_cols:
        out = test_df[["season", "home_team", "away_team", LABEL]].reset_index(drop=True)
        for col, vals in proba_cols.items():
            out[col] = vals
        out.to_csv(REPORTS_DIR / "proba_test.csv", index=False)
    print(f"  [guardado parcial en {REPORTS_DIR.name}/]")


def record(key, estimator, params, display_name):
    best_params[key] = params
    results.append(evaluate(estimator, X_test, y_test, display_name))
    proba = estimator.predict_proba(X_test)
    tabla_calibracion(proba, y_test, display_name)
    for i, c in enumerate(CLASSES):
        proba_cols[f"{key}_P{c}"] = proba[:, i]
    save_progress()


# =========================================
# 3. RANDOM FOREST
# =========================================
print("\n" + "=" * 60 + "\nRANDOM FOREST\n" + "=" * 60)
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid={
        "n_estimators": [200, 500],
        "max_depth": [2, 3, 4, 6, 8],
        "min_samples_leaf": [1, 4, 16, 64],
        "class_weight": [None, "balanced"],
    },
    cv=tscv, n_jobs=-1, scoring="neg_log_loss", refit=True, verbose=2,
)
rf_grid.fit(X_train, y_train)
print("Mejores hiperparámetros RF:", rf_grid.best_params_)
record("rf", rf_grid.best_estimator_, rf_grid.best_params_,
       f"Random Forest (test {TEST_SEASON})")

# =========================================
# 4. XGBOOST
# =========================================
print("\n" + "=" * 60 + "\nXGBOOST\n" + "=" * 60)
xgb_grid = GridSearchCV(
    estimator=XGBClassifierWithStringLabels(random_state=RANDOM_STATE),
    param_grid={
        "n_estimators": [200, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.1],
        "class_weight": [None, "balanced"],
    },
    cv=tscv, n_jobs=-1, scoring="neg_log_loss", refit=True, verbose=2,
)
xgb_grid.fit(X_train, y_train)
print("Mejores hiperparámetros XGB:", xgb_grid.best_params_)
best_xgb = xgb_grid.best_estimator_
assert list(best_xgb.classes_) == CLASSES, "Orden de clases inesperado en XGBoost"
record("xgb", best_xgb, xgb_grid.best_params_, f"XGBoost (test {TEST_SEASON})")

# =========================================
# 5. RED NEURONAL (Keras / TensorFlow)
# =========================================
print("\n" + "=" * 60 + "\nRED NEURONAL (Keras / TensorFlow)\n" + "=" * 60)
print("(importando TensorFlow, esto tarda ~1 min sin imprimir nada)")
from tensorflow import keras  # noqa: E402
from scikeras.wrappers import KerasClassifier  # noqa: E402
from keras_model_builder import build_keras_model  # noqa: E402
print("TensorFlow listo, arrancando la grilla.")

# EarlyStopping: antes eran 100 épocas fijas por combinación, siguiera
# mejorando o no. validation_split toma el ÚLTIMO 15% de cada fold, que en
# serie de tiempo es lo correcto (valida con lo más reciente, no con el futuro).
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)

keras_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KerasClassifier(
        model=build_keras_model,
        hidden_layer_sizes=(16,), activation="relu", alpha=0.0001,
        dropout=0.0, batch_size=32,
        epochs=200, callbacks=[early_stop], fit__validation_split=0.15,
        verbose=0, random_state=RANDOM_STATE)),
])

keras_grid = GridSearchCV(
    estimator=keras_pipeline,
    param_grid={
        "clf__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
        "clf__activation": ["relu", "linear"],
        "clf__alpha": [0.0001, 0.001, 0.005],
        "clf__dropout": [0.0, 0.2],
        "clf__batch_size": [32, 128],
    },
    cv=tscv, n_jobs=1,  # TensorFlow no es seguro con n_jobs=-1
    scoring="neg_log_loss", refit=True, verbose=2,
)
keras_grid.fit(X_train, y_train)
best_keras_params = {k.split("__")[-1]: v for k, v in keras_grid.best_params_.items()}
print("Mejores hiperparámetros Red Neuronal (Keras):", best_keras_params)
record("mlp_keras", keras_grid.best_estimator_, best_keras_params,
       f"Red Neuronal Keras (test {TEST_SEASON})")

# =========================================
# 6. COMPARATIVA FINAL
# =========================================
print("\n" + "=" * 60)
print(f"COMPARATIVA FINAL (test = {TEST_SEASON}) — ordenada por log-loss")
print("=" * 60)
print(pd.DataFrame(results).sort_values("log_loss").to_string(index=False))
print(f"\nReferencia: predecir siempre la tasa base -> log-loss {BASE_LOGLOSS:.4f}")
save_progress()
print("\n>>> Siguiente paso: python src/04_retrain_final_model.py")
