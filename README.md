# Predictor Premier League

Predice la probabilidad de victoria local / empate / victoria visitante para
cualquier partido de Premier League, usando ELO (K=30, acumulativo desde
2006-07) y win rate de los últimos 5 partidos como features.

## Estructura del proyecto

```
proyecto_final/
├── data/
│   ├── raw/                  <- CSVs originales (histórico + 2024/25-2025/26)
│   └── processed/            <- generados por 01 y 02 (bases unificadas, features, ELOs)
├── models/                   <- generado por 04 (modelo final .pkl + metadata)
├── reports/                  <- generado por 03 (comparación de modelos)
├── src/
│   ├── 01_unify_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_compare_models.py
│   └── 04_retrain_final_model.py
├── app/
│   ├── app.py                <- la app de Streamlit
│   ├── team_colors.py        <- colores por equipo para la barra de probabilidades
│   ├── team_badges.py        <- iniciales por equipo para la ficha circular (sin logos con derechos)
│   └── assets/
│       └── style.css         <- identidad visual (colores, tipografías Oswald/Inter)
├── .streamlit/
│   └── config.toml           <- tema oscuro nativo de Streamlit
├── requirements.txt
└── README.md
```

## Cómo correrlo (en orden)

```bash
# 1. Crear entorno e instalar dependencias
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Unificar la base histórica con 2024/25 y 2025/26
python src/01_unify_data.py

# 3. Calcular ELO y win_rate_5 sobre toda la base
python src/02_feature_engineering.py

# 4. Comparar Random Forest vs XGBoost vs Red Neuronal
#    (train hasta 2024-25, test en 2025-26)
python src/03_train_compare_models.py

# 5. Reentrenar el modelo GANADOR con el 100% de los datos
python src/04_retrain_final_model.py

# 6. Levantar la app
streamlit run app/app.py
```

## Decisiones de diseño (para recordar el "por qué")

- **ELO acumulativo, sin regresión a la media entre temporadas**: el rating
  de un equipo al cierre de una temporada es el punto de partida de la
  siguiente. Equipos nuevos/ascendidos arrancan en 1500.
- **`elo_diff` y `winrate_diff`** (no los valores absolutos) son las dos
  únicas features del modelo — así queda igual de simple que el proyecto
  original, con la ventaja de que el modelo no necesita saber "quién es
  quién", solo la diferencia relativa entre los dos equipos.
- **`home_winrate_5` / `away_winrate_5` son independientes entre sí**: el
  winrate de local de un equipo se calcula SOLO sobre sus últimos 5 partidos
  como local (no se mezcla con sus partidos de visitante), y viceversa.
- **Split temporal para evaluación**: entrenamos hasta 2024-25 y evaluamos en
  2025-26 (ya jugada por completo, así que es un backtest histórico). El
  modelo que gana esa comparación se vuelve a entrenar con el 100% de los
  datos (incluyendo 2025-26) para desplegar — mismos hiperparámetros, más
  datos.
- **Selección de equipos por dropdown** en la app (no texto libre) para
  evitar problemas de escritura/mayúsculas. Internamente todo se compara en
  mayúscula sostenida por seguridad extra.
- **Pipeline con `StandardScaler` siempre incluido**: no afecta a modelos de
  árboles (RF/XGBoost) y es necesario para la red neuronal — así `app.py` no
  necesita saber qué modelo ganó, simplemente llama `pipeline.predict_proba()`.

## Actualizar con partidos nuevos (temporada 2026-27 en adelante)

Cuando quieras que la app "aprenda" de partidos jugados después de 2025-26:
1. Agrega esos partidos a `data/raw/` con el mismo esquema.
2. Vuelve a correr `01 -> 02 -> 03 -> 04` en orden.
3. Los archivos en `data/processed/` (ELO y forma reciente) y `models/` se
   regeneran automáticamente con la información más actualizada.
