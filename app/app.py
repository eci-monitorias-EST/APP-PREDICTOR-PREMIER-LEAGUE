"""
app.py
App de Streamlit: el usuario elige equipo local y visitante, y la app
muestra la probabilidad de victoria local / empate / victoria visitante
usando el modelo entrenado en src/04_retrain_final_model.py.

Ejecutar con: streamlit run app/app.py   (desde la raíz del proyecto)
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from team_colors import get_team_color, DEFAULT_COLOR
from team_badges import render_team_badge

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="Predictor Premier League", page_icon="⚽", layout="centered")


def load_css():
    """Inyecta app/assets/style.css. Es solo presentación: si el archivo
    no existe por alguna razón, la app sigue funcionando con el estilo
    por defecto de Streamlit."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()


# =========================================
# CARGA DE ARTEFACTOS (una sola vez, cacheado)
# =========================================
@st.cache_resource
def load_model():
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    if metadata.get("is_keras", False):
        # Modelo de Keras/TensorFlow: se guardó distinto (formato nativo .keras
        # + scaler aparte), así que armamos un pequeño envoltorio para que el
        # resto de app.py pueda seguir llamando pipeline.predict_proba(X)
        # exactamente igual que con los modelos de scikit-learn.
        import tensorflow as tf

        scaler = joblib.load(MODELS_DIR / "keras_scaler.pkl")
        keras_model = tf.keras.models.load_model(MODELS_DIR / "keras_model.keras")

        class KerasPipelineWrapper:
            def predict_proba(self, X):
                X_scaled = scaler.transform(X)
                return keras_model.predict(X_scaled, verbose=0)

        pipeline = KerasPipelineWrapper()
    else:
        pipeline = joblib.load(MODELS_DIR / "modelo_final.pkl")

    return pipeline, metadata


@st.cache_data
def load_team_state():
    with open(PROCESSED_DIR / "elo_ratings_final.json") as f:
        elo_ratings = json.load(f)
    with open(PROCESSED_DIR / "team_home_form.json") as f:
        home_form = json.load(f)
    with open(PROCESSED_DIR / "team_away_form.json") as f:
        away_form = json.load(f)
    return elo_ratings, home_form, away_form


pipeline, metadata = load_model()
elo_ratings, home_form, away_form = load_team_state()
CLASSES_ORDER = metadata["classes_order"]  # ej. ['A', 'D', 'H']

# Normalizamos nombres para evitar problemas de mayúsculas/espacios,
# tal como se acordó: comparación siempre en MAYÚSCULA SOSTENIDA.
NORMALIZED_TO_ORIGINAL = {team.strip().upper(): team for team in elo_ratings.keys()}
TEAM_LIST = sorted(elo_ratings.keys())


def normalize_team(name: str) -> str:
    """Devuelve el nombre canónico del dataset a partir de cualquier variante
    de mayúsculas/espacios que escriba el usuario. Si no existe, regresa None."""
    return NORMALIZED_TO_ORIGINAL.get(name.strip().upper())


def get_features(home_team: str, away_team: str) -> pd.DataFrame:
    home_elo = elo_ratings.get(home_team, 1500)
    away_elo = elo_ratings.get(away_team, 1500)

    home_hist = home_form.get(home_team, [])
    away_hist = away_form.get(away_team, [])
    home_wr = float(np.mean(home_hist)) if home_hist else 0.0
    away_wr = float(np.mean(away_hist)) if away_hist else 0.0

    elo_diff = home_elo - away_elo
    winrate_diff = home_wr - away_wr

    return pd.DataFrame([[elo_diff, winrate_diff]], columns=metadata["features"])


def predict_probabilities(home_team: str, away_team: str):
    X = get_features(home_team, away_team)
    proba = pipeline.predict_proba(X)[0]
    proba_dict = dict(zip(CLASSES_ORDER, proba))
    return {
        "home_win": proba_dict.get("H", 0.0),
        "draw": proba_dict.get("D", 0.0),
        "away_win": proba_dict.get("A", 0.0),
    }


def render_probability_bar(home_team, away_team, probs):
    home_color = get_team_color(home_team)
    away_color = get_team_color(away_team)

    p_home = probs["home_win"] * 100
    p_draw = probs["draw"] * 100
    p_away = probs["away_win"] * 100

    html = f"""
    <div class="result-panel">
      <div class="result-row">
        <div>
          {render_team_badge(home_team, home_color)}
          <div class="team-name">{home_team}</div>
          <div class="result-pct" style="color:{home_color};">{p_home:.0f}%</div>
        </div>
        <div>
          <div class="draw-label">Empate</div>
          <div class="result-pct result-pct--draw">{p_draw:.0f}%</div>
        </div>
        <div>
          {render_team_badge(away_team, away_color)}
          <div class="team-name">{away_team}</div>
          <div class="result-pct" style="color:{away_color};">{p_away:.0f}%</div>
        </div>
      </div>
      <div class="prob-track">
        <div style="width:{p_home}%; background-color:{home_color};"></div>
        <div style="width:{p_draw}%; background-color:#5C6F63;"></div>
        <div style="width:{p_away}%; background-color:{away_color};"></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================================
# UI
# =========================================
st.title("⚽ Predictor Premier League")
st.caption(
    f"Modelo desplegado: **{metadata['model_name']}** · "
    f"entrenado con {metadata['trained_rows']} partidos (hasta {metadata['trained_through_season']})"
)

with st.container(border=True):
    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        home_team = st.selectbox(
            "Equipo local", TEAM_LIST,
            index=TEAM_LIST.index("Arsenal") if "Arsenal" in TEAM_LIST else 0,
        )
        st.markdown(render_team_badge(home_team, get_team_color(home_team)), unsafe_allow_html=True)
    with col_vs:
        st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)
    with col2:
        away_team = st.selectbox(
            "Equipo visitante", TEAM_LIST,
            index=TEAM_LIST.index("Liverpool") if "Liverpool" in TEAM_LIST else 1,
        )
        st.markdown(render_team_badge(away_team, get_team_color(away_team)), unsafe_allow_html=True)

if home_team == away_team:
    st.warning("Elige dos equipos distintos.")
else:
    if st.button("Predecir resultado", type="primary"):
        probs = predict_probabilities(home_team, away_team)
        st.divider()
        render_probability_bar(home_team, away_team, probs)

        with st.expander("Ver detalle (ELO y forma reciente usados)"):
            home_elo = elo_ratings.get(home_team, 1500)
            away_elo = elo_ratings.get(away_team, 1500)
            st.write(f"**ELO {home_team}:** {home_elo:.1f}  |  **ELO {away_team}:** {away_elo:.1f}")
            home_hist = home_form.get(home_team, [])
            away_hist = away_form.get(away_team, [])
            st.write(
                f"**% victorias últimos {len(home_hist)} de local ({home_team}):** "
                f"{np.mean(home_hist)*100:.0f}%" if home_hist else f"Sin historial de local para {home_team}"
            )
            st.write(
                f"**% victorias últimos {len(away_hist)} de visitante ({away_team}):** "
                f"{np.mean(away_hist)*100:.0f}%" if away_hist else f"Sin historial de visitante para {away_team}"
            )

st.divider()
st.caption(
    "Nota: el ELO y la forma reciente reflejan el estado de los equipos al cierre de la "
    "temporada 2025-2026. Si juegan más partidos después de esa fecha, hay que "
    "actualizar estos datos (re-correr el pipeline con los resultados nuevos) para que "
    "la predicción siga siendo precisa."
)