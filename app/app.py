"""
app.py
App de Streamlit con dos pestañas:

  1. "Predecir un partido": el usuario elige local y visitante, y la app
     muestra la probabilidad de victoria local / empate / victoria visitante
     usando el modelo entrenado en src/04_retrain_final_model.py.

  2. "Temporada 2026-2027": resultados de la simulación de Monte Carlo que
     genera src/05_predict_next_season.py — tabla de posiciones esperada con
     su incertidumbre, y el detalle partido a partido de los 380 encuentros.

Ejecutar con: streamlit run app/app.py   (desde la raíz del proyecto)
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from team_colors import get_team_color, DEFAULT_COLOR
from team_badges import render_team_badge

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="Predictor Premier League", layout="centered")


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


TEMPORADA_SIM = "2026_2027"


@st.cache_data
def load_simulacion():
    """Lee los CSV que produce src/05_predict_next_season.py.

    Devuelve (partidos, tabla) o (None, None) si todavía no se ha corrido la
    simulación. La app tiene que seguir funcionando sin ellos: la pestaña 1 no
    los necesita para nada.
    """
    f_part = PROCESSED_DIR / f"prediccion_partidos_{TEMPORADA_SIM}.csv"
    f_tab = PROCESSED_DIR / f"prediccion_tabla_{TEMPORADA_SIM}.csv"
    if not (f_part.exists() and f_tab.exists()):
        return None, None
    return pd.read_csv(f_part), pd.read_csv(f_tab)


@st.cache_data
def load_simulaciones_completas():
    """Carga el .npz con las 10.000 temporadas simuladas (~1 MB comprimido).

    Devuelve None si no existe: la app funciona igual, solo que sin el
    explorador de simulaciones individuales.
    """
    f = PROCESSED_DIR / f"simulaciones_{TEMPORADA_SIM}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    puntos = z["puntos"]
    # posicion[s, t] = puesto del equipo t en la simulacion s (1 = campeon)
    orden = (-puntos).argsort(axis=1)
    posicion = np.empty_like(orden)
    np.put_along_axis(posicion, orden,
                      np.arange(1, puntos.shape[1] + 1)[None, :].repeat(puntos.shape[0], 0), axis=1)
    return {
        "resultados": z["resultados"], "puntos": puntos, "posicion": posicion,
        "equipos": list(z["equipos"]), "clases": list(z["clases"]),
        "destacadas": list(z["destacadas"]), "etiquetas": list(z["etiquetas"]),
    }


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
st.title("Predictor Premier League")
st.caption(
    f"Modelo desplegado: **{metadata['model_name']}** · "
    f"entrenado con {metadata['trained_rows']} partidos (hasta {metadata['trained_through_season']})"
)

tab_partido, tab_temporada = st.tabs(["Predecir un partido", "Temporada 2026-2027"])

with tab_partido:
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


# =========================================
# PESTAÑA 2: TEMPORADA SIMULADA
# =========================================
NOMBRES_DESTACADAS = {
    "representativa": "Representativa (típica en todo)",
    "mas_parecida_a_lo_esperado": "La más parecida a lo esperado",
    "mas_caotica": "La más caótica",
}

with tab_temporada:
    partidos_sim, tabla_sim = load_simulacion()

    if partidos_sim is None:
        st.info(
            "Todavía no hay simulación. Genérala con:\n\n"
            "```\npython src/05_predict_next_season.py\n```\n\n"
            "Necesita `data/raw/calendario_2026_2027.csv` y el modelo de `models/`."
        )
    else:
        sims = load_simulaciones_completas()
        n_sims = sims["puntos"].shape[0] if sims else 10_000

        st.caption(
            f"Temporada 2026-2027 simulada {n_sims:,} veces. En cada simulación se recorren "
            "los 380 partidos en orden cronológico, se sortea el resultado según las "
            "probabilidades del modelo y se recalcula el ELO con ese resultado."
        )

        sub_resumen, sub_explorar, sub_partidos = st.tabs(
            ["Resumen de las simulaciones", "Explorar una temporada", "Partido a partido"])

        # -------------------------------------------------
        # RESUMEN
        # -------------------------------------------------
        with sub_resumen:
            campeon, colista = tabla_sim.iloc[0], tabla_sim.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Favorito al título", campeon["equipo"],
                      f"{campeon['prob_campeon'] * 100:.1f}% de las simulaciones",
                      delta_color="off")
            c2.metric("Puntos esperados del líder", f"{campeon['puntos_medios']:.0f}",
                      f"rango {campeon['puntos_p5']:.0f}–{campeon['puntos_p95']:.0f}",
                      delta_color="off")
            c3.metric("Mayor riesgo de descenso", colista["equipo"],
                      f"{colista['prob_descenso'] * 100:.1f}% de las simulaciones",
                      delta_color="off")

            st.divider()
            st.subheader("Tabla de posiciones esperada")
            st.caption(
                "Los puntos son el promedio entre simulaciones y el rango va del percentil 5 al "
                "95: el 90% de los desenlaces cae dentro de él. Ese rango es ancho a propósito, "
                "porque una sola temporada tiene mucha variabilidad. Ojo: promediar aplasta la "
                "tabla, así que esta NO es una temporada plausible — para eso está la pestaña "
                "'Explorar una temporada'."
            )
            vista = tabla_sim.copy()
            vista["rango"] = (vista["puntos_p5"].astype(int).astype(str) + "–"
                              + vista["puntos_p95"].astype(int).astype(str))
            for c in ("prob_campeon", "prob_top4", "prob_descenso"):
                vista[c] = vista[c] * 100
            st.dataframe(
                vista[["puesto_esperado", "equipo", "puntos_medios", "rango", "prob_campeon",
                       "prob_top4", "prob_descenso", "ganados_medios", "empatados_medios",
                       "perdidos_medios"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "puesto_esperado": st.column_config.NumberColumn("#", width="small"),
                    "equipo": st.column_config.TextColumn("Equipo"),
                    "puntos_medios": st.column_config.NumberColumn("Puntos", format="%.1f"),
                    "rango": st.column_config.TextColumn("Rango p5–p95"),
                    "prob_campeon": st.column_config.ProgressColumn(
                        "Campeón", format="%.1f%%", min_value=0, max_value=100),
                    "prob_top4": st.column_config.ProgressColumn(
                        "Top 4", format="%.1f%%", min_value=0, max_value=100),
                    "prob_descenso": st.column_config.ProgressColumn(
                        "Desciende", format="%.1f%%", min_value=0, max_value=100),
                    "ganados_medios": st.column_config.NumberColumn("G", format="%.1f"),
                    "empatados_medios": st.column_config.NumberColumn("E", format="%.1f"),
                    "perdidos_medios": st.column_config.NumberColumn("P", format="%.1f"),
                },
            )

            st.divider()
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Probabilidad de ser campeón**")
                d = tabla_sim.nlargest(10, "prob_campeon")[["equipo", "prob_campeon"]].copy()
                d["pct"] = d["prob_campeon"] * 100
                st.altair_chart(
                    alt.Chart(d).mark_bar(color="#7BD389").encode(
                        x=alt.X("pct:Q", title="% de simulaciones"),
                        y=alt.Y("equipo:N", sort="-x", title=None),
                        tooltip=[alt.Tooltip("equipo:N", title="Equipo"),
                                 alt.Tooltip("pct:Q", title="% campeón", format=".1f")],
                    ).properties(height=300), use_container_width=True)
            with g2:
                st.markdown("**Probabilidad de descender**")
                d = tabla_sim.nlargest(10, "prob_descenso")[["equipo", "prob_descenso"]].copy()
                d["pct"] = d["prob_descenso"] * 100
                st.altair_chart(
                    alt.Chart(d).mark_bar(color="#D96C6C").encode(
                        x=alt.X("pct:Q", title="% de simulaciones"),
                        y=alt.Y("equipo:N", sort="-x", title=None),
                        tooltip=[alt.Tooltip("equipo:N", title="Equipo"),
                                 alt.Tooltip("pct:Q", title="% descenso", format=".1f")],
                    ).properties(height=300), use_container_width=True)

            if sims:
                st.divider()
                st.markdown("**¿En qué puesto termina cada equipo?**")
                st.caption(
                    "Distribución del puesto final entre todas las simulaciones. Cuanto más "
                    "ancha la distribución, menos seguro está el modelo del destino del equipo."
                )
                eq_sel = st.selectbox("Equipo", sims["equipos"],
                                      index=sims["equipos"].index(campeon["equipo"]),
                                      key="eq_hist")
                pos = sims["posicion"][:, sims["equipos"].index(eq_sel)]
                d = (pd.Series(pos).value_counts().rename_axis("puesto")
                     .reset_index(name="veces").sort_values("puesto"))
                d["pct"] = d["veces"] / len(pos) * 100
                st.altair_chart(
                    alt.Chart(d).mark_bar(color="#6C9BD9").encode(
                        x=alt.X("puesto:O", title="Puesto final"),
                        y=alt.Y("pct:Q", title="% de simulaciones"),
                        tooltip=[alt.Tooltip("puesto:O", title="Puesto"),
                                 alt.Tooltip("pct:Q", title="%", format=".1f"),
                                 alt.Tooltip("veces:Q", title="Simulaciones")],
                    ).properties(height=260), use_container_width=True)
                st.caption(
                    f"{eq_sel}: puesto medio {pos.mean():.1f} · mejor {pos.min()} · "
                    f"peor {pos.max()} · termina entre los 4 primeros en "
                    f"{(pos <= 4).mean() * 100:.1f}% y entre los 3 últimos en "
                    f"{(pos >= 18).mean() * 100:.1f}% de las simulaciones."
                )

        # -------------------------------------------------
        # EXPLORAR UNA TEMPORADA
        # -------------------------------------------------
        with sub_explorar:
            if not sims:
                st.info(
                    "Falta `data/processed/simulaciones_2026_2027.npz`. "
                    "Vuelve a correr `python src/05_predict_next_season.py`."
                )
            else:
                st.caption(
                    "Cada simulación es una temporada completa y coherente: los 380 resultados "
                    "producen exactamente esa tabla. Elige una de las destacadas o escribe "
                    "cualquier número."
                )
                e1, e2 = st.columns([3, 2])
                with e1:
                    opciones = [NOMBRES_DESTACADAS.get(e, e) for e in sims["etiquetas"]] + \
                               ["Otra (escribir número)"]
                    elegida = st.radio("Temporada", opciones, horizontal=False, key="sel_temp")
                with e2:
                    if elegida == "Otra (escribir número)":
                        i_sim = st.number_input("Número de simulación", min_value=1,
                                                max_value=n_sims, value=1, step=1,
                                                key="num_sim") - 1
                    else:
                        i_sim = int(sims["destacadas"][opciones.index(elegida)])
                        st.metric("Simulación", f"#{i_sim + 1:,}")

                pts_s = sims["puntos"][i_sim]
                res_s = [sims["clases"][int(c)] for c in sims["resultados"][:, i_sim]]
                orden = np.argsort(-pts_s)

                t1, t2, t3 = st.columns(3)
                t1.metric("Campeón", sims["equipos"][orden[0]], f"{pts_s[orden[0]]} puntos",
                          delta_color="off")
                t2.metric("Empates", f"{res_s.count('D')}",
                          f"{res_s.count('D') / len(res_s) * 100:.1f}% de los partidos",
                          delta_color="off")
                t3.metric("Último", sims["equipos"][orden[-1]], f"{pts_s[orden[-1]]} puntos",
                          delta_color="off")

                st.divider()
                v1, v2 = st.columns([2, 3])
                with v1:
                    st.markdown("**Tabla final**")
                    tf = pd.DataFrame({
                        "#": range(1, len(orden) + 1),
                        "Equipo": [sims["equipos"][k] for k in orden],
                        "Pts": [int(pts_s[k]) for k in orden],
                        "Esperados": [float(tabla_sim.set_index("equipo").loc[
                            sims["equipos"][k], "puntos_medios"]) for k in orden],
                    })
                    tf["Dif"] = (tf["Pts"] - tf["Esperados"]).round(1)
                    st.dataframe(tf, hide_index=True, use_container_width=True, height=560,
                                 column_config={
                                     "#": st.column_config.NumberColumn(width="small"),
                                     "Esperados": st.column_config.NumberColumn(format="%.1f"),
                                     "Dif": st.column_config.NumberColumn(
                                         format="%+.1f",
                                         help="Puntos por encima (+) o por debajo (−) de lo esperado"),
                                 })
                with v2:
                    st.markdown("**Puntos en esta temporada vs. los esperados**")
                    d = pd.DataFrame({
                        "equipo": [sims["equipos"][k] for k in orden],
                        "Esta temporada": [int(pts_s[k]) for k in orden],
                        "Promedio de las simulaciones": [
                            float(tabla_sim.set_index("equipo").loc[
                                sims["equipos"][k], "puntos_medios"]) for k in orden],
                    }).melt("equipo", var_name="serie", value_name="puntos")
                    st.altair_chart(
                        alt.Chart(d).mark_bar().encode(
                            x=alt.X("puntos:Q", title="Puntos"),
                            y=alt.Y("equipo:N", sort=[sims["equipos"][k] for k in orden],
                                    title=None),
                            color=alt.Color("serie:N", title=None,
                                            scale=alt.Scale(range=["#6C9BD9", "#8A8A8A"]),
                                            legend=alt.Legend(orient="top")),
                            yOffset="serie:N",
                            tooltip=["equipo:N", "serie:N", "puntos:Q"],
                        ).properties(height=560), use_container_width=True)

                st.divider()
                st.markdown("**Los 380 partidos de esta temporada**")
                det = partidos_sim[["fecha", "home_team", "away_team"]].copy()
                det["resultado"] = res_s
                det["texto"] = np.where(det.resultado == "H", "Gana local",
                                        np.where(det.resultado == "D", "Empate", "Gana visitante"))
                f1, f2 = st.columns([2, 3])
                with f1:
                    eqs = ["Todos"] + sims["equipos"]
                    fe = st.selectbox("Filtrar por equipo", eqs, key="f_eq_sim")
                with f2:
                    fechas = sorted(det.fecha.unique())
                    fr = st.select_slider("Rango de fechas", options=fechas,
                                          value=(fechas[0], fechas[-1]), key="f_fecha_sim")
                d2 = det[(det.fecha >= fr[0]) & (det.fecha <= fr[1])]
                if fe != "Todos":
                    d2 = d2[(d2.home_team == fe) | (d2.away_team == fe)]
                st.caption(f"{len(d2)} partidos")
                st.dataframe(d2[["fecha", "home_team", "away_team", "texto"]],
                             hide_index=True, use_container_width=True, height=400,
                             column_config={
                                 "fecha": st.column_config.TextColumn("Fecha", width="small"),
                                 "home_team": st.column_config.TextColumn("Local"),
                                 "away_team": st.column_config.TextColumn("Visitante"),
                                 "texto": st.column_config.TextColumn("Resultado"),
                             })

                with st.expander("¿Cuál de estas temporadas debería usar en el informe?"):
                    st.markdown(
                        "**Representativa** — la más típica: sus puntos de campeón, de colista, "
                        "su dispersión, su número de empates y su coincidencia con el orden "
                        "esperado están todos cerca de la mediana. Es la más honesta como "
                        "\"así podría verse la temporada\".\n\n"
                        "**La más parecida a lo esperado** — la que mejor reproduce el orden del "
                        "promedio (Arsenal, City, United...). Es la más limpia de presentar, "
                        "pero justamente por eso es POCO típica: casi no tiene sorpresas, y una "
                        "temporada real siempre las trae.\n\n"
                        "**La más caótica** — el otro extremo, donde el orden final casi no se "
                        "parece al esperado. Sirve para mostrar hasta dónde llega la "
                        "incertidumbre.\n\n"
                        "Si vas a mostrar una sola, usa la representativa y menciona que existen "
                        "las otras dos. Si quieres contrastar, poner la esperada al lado de la "
                        "caótica deja clarísimo el mensaje: el modelo ordena bien en promedio, "
                        "pero una temporada concreta puede alejarse muchísimo."
                    )

        # -------------------------------------------------
        # PARTIDO A PARTIDO (probabilidades)
        # -------------------------------------------------
        with sub_partidos:
            st.caption(
                "La predicción real del modelo para cada partido son estas tres "
                "probabilidades, promediadas entre todas las simulaciones."
            )
            f1, f2 = st.columns([2, 3])
            with f1:
                equipos_sim = ["Todos"] + sorted(
                    set(partidos_sim.home_team) | set(partidos_sim.away_team))
                filtro_equipo = st.selectbox("Filtrar por equipo", equipos_sim, key="f_eq_prob")
            with f2:
                fechas = sorted(partidos_sim.fecha.unique())
                rango_fechas = st.select_slider("Rango de fechas", options=fechas,
                                                value=(fechas[0], fechas[-1]), key="f_fecha_prob")

            vista_p = partidos_sim[(partidos_sim.fecha >= rango_fechas[0])
                                   & (partidos_sim.fecha <= rango_fechas[1])]
            if filtro_equipo != "Todos":
                vista_p = vista_p[(vista_p.home_team == filtro_equipo)
                                  | (vista_p.away_team == filtro_equipo)]
            vista_p = vista_p.copy()
            for c in ("prob_local", "prob_empate", "prob_visitante"):
                vista_p[c] = vista_p[c] * 100

            st.caption(f"{len(vista_p)} partidos")
            st.dataframe(
                vista_p[["fecha", "home_team", "away_team", "prob_local", "prob_empate",
                         "prob_visitante"]],
                hide_index=True, use_container_width=True, height=440,
                column_config={
                    "fecha": st.column_config.TextColumn("Fecha", width="small"),
                    "home_team": st.column_config.TextColumn("Local"),
                    "away_team": st.column_config.TextColumn("Visitante"),
                    "prob_local": st.column_config.ProgressColumn(
                        "Gana local", format="%.1f%%", min_value=0, max_value=100),
                    "prob_empate": st.column_config.ProgressColumn(
                        "Empate", format="%.1f%%", min_value=0, max_value=100),
                    "prob_visitante": st.column_config.ProgressColumn(
                        "Gana visitante", format="%.1f%%", min_value=0, max_value=100),
                },
            )

            with st.expander("¿Por qué el empate nunca es el resultado más probable?"):
                st.markdown(
                    "Para que lo fuera haría falta P(empate) > 1/3: si P(empate) ≤ 1/3, "
                    "entonces P(local) + P(visitante) ≥ 2/3 y una de las dos siempre le gana. "
                    "Como en la Premier se empata alrededor del 26% de los partidos, un modelo "
                    "bien calibrado casi nunca supera ese 1/3. No es un defecto del modelo: es "
                    "consecuencia de que sus probabilidades sean correctas.\n\n"
                    "Por eso la temporada se **simula sorteando** cada resultado en vez de "
                    "quedarse con el más probable. Así los empates aparecen en la proporción "
                    "que les corresponde (~25% de los partidos simulados) y la tabla reparte "
                    "unos 1.035 puntos, como una temporada real, en vez de los 1.140 que "
                    "saldrían si nunca hubiera empates."
                )
