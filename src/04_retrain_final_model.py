"""
04_retrain_final_model.py

Etapa B del plan acordado:
  - Lee reports/model_comparison_results.csv (generado por 03) y elige
    automáticamente el modelo con mejor ROC-AUC macro.
  - Toma sus hiperparámetros ganadores desde reports/best_hyperparams.json.
  - Reentrena ese modelo con el 100% de los datos (2006-07 -> 2025-26),
    usando los MISMOS hiperparámetros (sin volver a tunear).
  - Guarda el modelo final en models/modelo_final.pkl junto con sus metadatos.

Este es el modelo que la app de Streamlit carga para predecir partidos.

Ejecutar desde cualquier lado con: python src/04_retrain_final_model.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 2658
FEATURES = ["elo_diff", "winrate_diff"]
LABEL = "result"

# =========================================
# 1. ELEGIR EL MEJOR MODELO SEGÚN 03_train_compare_models.py
# =========================================
results_df = pd.read_csv(REPORTS_DIR / "model_comparison_results.csv")
# Mismo criterio que 03: gana el menor log-loss. El log-loss es una regla de
# puntuacion propia (su minimo esta en la probabilidad verdadera), asi que es
# la metrica correcta cuando el entregable son las probabilidades que muestra
# la app. Ordenar por ROC-AUC aqui podria elegir un modelo distinto al que 03
# declaro ganador.
best_row = results_df.sort_values("log_loss", ascending=True).iloc[0]
best_model_name = best_row["model"]
print(f"Modelo ganador según reports/model_comparison_results.csv:\n  -> {best_model_name}")
print(f"  Log-loss = {best_row['log_loss']:.4f} (criterio de seleccion) | "
      f"ROC-AUC macro = {best_row['roc_auc_macro']:.4f}")

with open(REPORTS_DIR / "best_hyperparams.json") as f:
    best_hyperparams = json.load(f)

if "Random Forest" in best_model_name:
    model_key = "rf"
elif "XGBoost" in best_model_name:
    model_key = "xgb"
elif "Keras" in best_model_name:
    model_key = "mlp_keras"
elif "MLP" in best_model_name or "Red neuronal" in best_model_name or "Red Neuronal" in best_model_name:
    model_key = "mlp"
else:
    raise ValueError(f"No reconozco el modelo '{best_model_name}' - revisa el nombre en el CSV.")

hyperparams = best_hyperparams[model_key]
print(f"Hiperparámetros a usar ({model_key}):", hyperparams)

# =========================================
# 2. CONSTRUIR EL PIPELINE FINAL
#    (siempre con StandardScaler adentro: no afecta a RF/XGB y es necesario
#     para el MLP -> así la app.py no necesita saber qué modelo ganó)
# =========================================
if model_key == "rf":
    clf = RandomForestClassifier(random_state=RANDOM_STATE, **hyperparams)
elif model_key == "xgb":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from xgb_wrapper import XGBClassifierWithStringLabels
    clf = XGBClassifierWithStringLabels(random_state=RANDOM_STATE, **hyperparams)
elif model_key == "mlp":
    clf = MLPClassifier(
        max_iter=1000, early_stopping=False, random_state=RANDOM_STATE, **hyperparams,
    )
elif model_key == "mlp_keras":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tensorflow import keras
    from scikeras.wrappers import KerasClassifier
    from keras_model_builder import build_keras_model

    # El modelo final tiene que entrenarse IGUAL que el que gano la comparacion
    # en 03. Alli la seleccion se hizo con epochs=200 + EarlyStopping; si aqui
    # se entrenara con 100 epocas fijas, el modelo desplegado no seria el que
    # se midio (con batch_size grande y SGD lento, se quedaria corto).
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=0,
    )
    clf = KerasClassifier(
        model=build_keras_model,
        epochs=200,
        callbacks=[early_stop],
        fit__validation_split=0.15,
        verbose=0,
        random_state=RANDOM_STATE,
        **hyperparams,  # hidden_layer_sizes, activation, alpha, dropout, batch_size
    )

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", clf),
])

# =========================================
# 3. REENTRENAR CON EL 100% DE LOS DATOS (2006-07 -> 2025-26)
# =========================================
df = pd.read_csv(PROCESSED_DIR / "dfpremier_features.csv")
X, y = df[FEATURES], df[LABEL]

pipeline.fit(X, y)
classes_order = list(pipeline.named_steps["clf"].classes_)
print("Orden de clases del modelo final:", classes_order)
assert classes_order == sorted(classes_order), "Las clases no quedaron en orden alfabético A/D/H"

# =========================================
# 4. GUARDAR MODELO + METADATOS
#    Los modelos de scikit-learn (RF, XGBoost, MLP sklearn) se guardan todos
#    juntos en un .pkl con joblib. Los modelos de Keras/TensorFlow NO se
#    deben guardar así (es frágil entre versiones) - Keras tiene su propio
#    formato nativo (.keras), así que ahí guardamos el scaler y la red por
#    separado. app.py revisa "is_keras" en los metadatos para saber cuál de
#    los dos casos cargar.
# =========================================
is_keras = (model_key == "mlp_keras")

if is_keras:
    joblib.dump(pipeline.named_steps["scaler"], MODELS_DIR / "keras_scaler.pkl")
    pipeline.named_steps["clf"].model_.save(MODELS_DIR / "keras_model.keras")
    # Si existía un modelo sklearn de una corrida anterior, lo borramos para
    # que no queden artefactos viejos confundiendo a la app.
    (MODELS_DIR / "modelo_final.pkl").unlink(missing_ok=True)
else:
    joblib.dump(pipeline, MODELS_DIR / "modelo_final.pkl")
    (MODELS_DIR / "keras_scaler.pkl").unlink(missing_ok=True)
    (MODELS_DIR / "keras_model.keras").unlink(missing_ok=True)

metadata = {
    "model_key": model_key,
    "model_name": best_model_name,
    "features": FEATURES,
    "classes_order": classes_order,  # ej. ['A', 'D', 'H']
    "hyperparams": hyperparams,
    "trained_rows": len(df),
    "trained_through_season": "2025-2026",
    "is_keras": is_keras,
}
with open(MODELS_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

if is_keras:
    print(f"\nGuardado: {MODELS_DIR}/keras_model.keras, keras_scaler.pkl y model_metadata.json")
else:
    print(f"\nGuardado: {MODELS_DIR}/modelo_final.pkl y model_metadata.json")
print(">>> Siguiente paso: streamlit run app/app.py")