"""
05_predict_next_season.py

Simula la temporada 2026-2027 completa con el modelo final (el que dejó
04_retrain_final_model.py) y el calendario oficial.

POR QUÉ SE SIMULA Y NO SE "PREDICE"
-----------------------------------
La forma obvia sería recorrer los 380 partidos y quedarse con la clase más
probable (argmax), como en el notebook original. Eso NO sirve con este modelo,
por dos razones:

1) El argmax nunca elige empate. Para que lo hiciera haría falta P(D) > 1/3:
   si P(D) <= 1/3, entonces P(H)+P(A) >= 2/3 y una de las dos le gana siempre.
   Como el empate ocurre ~26% de las veces, un modelo bien calibrado nunca
   pasa de 1/3. Resultado: los 380 partidos se repartirían 1140 puntos en vez
   de los ~1035 reales y todos los equipos terminarían con cero empates.

2) Los errores se acumulan. Si el ELO se actualiza con el resultado predicho y
   el predicho es siempre el favorito, los fuertes ganan ELO sin parar y los
   débiles lo pierden sin parar. Es un lazo de retroalimentación: en la versión
   del notebook el campeón predicho terminó con 110 puntos contra los 84 del
   campeón real.

La solución es SORTEAR el resultado según las probabilidades del modelo en vez
de tomar el máximo, y repetir la temporada muchas veces:

- Los empates aparecen solos. Si el modelo dice P(D) = 0.26, en el 26% de las
  simulaciones ese partido termina en empate, aunque el argmax jamás lo elija.
- No hay divergencia sistemática: cada simulación toma un camino distinto y el
  favorito gana la mayoría de las veces, pero no todas.
- En vez de una tabla inventada se obtiene una DISTRIBUCIÓN: puntos esperados
  con su rango, probabilidad de ser campeón, de entrar a top-4, de descender.

Esto es válido porque las probabilidades están bien calibradas (ver el
log-loss de reports/model_comparison_results.csv). Sortear de una distribución
mal calibrada propagaría el sesgo a toda la simulación.

CÓMO SE HACE RÁPIDO
-------------------
10.000 simulaciones x 380 partidos son 3,8 millones de evaluaciones. Una por
una, con Keras, serían horas. Pero el bucle se puede voltear: en el partido i
TODAS las simulaciones están en el mismo enfrentamiento, solo con ELOs
distintos. Entonces se arman las 10.000 filas de features y se hace UNA sola
llamada a predict_proba con el lote completo. Quedan 380 llamadas en vez de
3,8 millones.

ESTADO INICIAL
--------------
Sale de data/processed/, tal como lo dejó 02_feature_engineering.py:
  - elo_ratings_final.json  -> ELO de cada equipo al cierre de 2025-2026
  - team_home_form.json     -> últimos 5 resultados como local
  - team_away_form.json     -> últimos 5 resultados como visitante
Los tres YA incluyen el último partido de la temporada. Un equipo sin historia
(un ascendido que nunca estuvo en la Premier) arranca en ELO 1500, que en esta
escala queda alrededor del puesto 17 de 20 — un valor razonable para un recién
ascendido, y que además se ajusta solo a medida que avanza la simulación.

Ejecutar desde cualquier lado con: python src/05_predict_next_season.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"

N_SIMS = 10_000           # temporadas simuladas
K_ELO = 30                # mismo k que 02_feature_engineering.py
WINDOW = 5                # mismos últimos 5 partidos
SEED = 2658
TEMPORADA = "2026-2027"

rng = np.random.default_rng(SEED)

# =========================================
# 1. MODELO (misma lógica de carga que app.py)
# =========================================
with open(MODELS_DIR / "model_metadata.json") as f:
    META = json.load(f)
CLASSES = META["classes_order"]          # ['A', 'D', 'H']
FEATURES = META["features"]              # ['elo_diff', 'winrate_diff']
iA, iD, iH = CLASSES.index("A"), CLASSES.index("D"), CLASSES.index("H")

if META.get("is_keras", False):
    import joblib
    import tensorflow as tf
    scaler = joblib.load(MODELS_DIR / "keras_scaler.pkl")
    keras_model = tf.keras.models.load_model(MODELS_DIR / "keras_model.keras")

    def predict_proba(X):
        return keras_model.predict(scaler.transform(X), verbose=0, batch_size=8192)
else:
    import joblib
    pipeline = joblib.load(MODELS_DIR / "modelo_final.pkl")

    def predict_proba(X):
        return pipeline.predict_proba(X)

print(f"Modelo: {META['model_name']}")
print(f"Entrenado con {META['trained_rows']} partidos hasta {META['trained_through_season']}")

# =========================================
# 2. CALENDARIO Y ESTADO INICIAL
# =========================================
cal = pd.read_csv(RAW_DIR / f"calendario_{TEMPORADA.replace('-', '_')}.csv")
cal = cal.reset_index(drop=True)

equipos = sorted(set(cal.home_team) | set(cal.away_team))
idx = {t: i for i, t in enumerate(equipos)}
T, M = len(equipos), len(cal)

# --- validación del calendario: un error aquí arruina toda la simulación ---
assert M == 380, f"El calendario tiene {M} partidos, deberían ser 380"
assert T == 20, f"El calendario tiene {T} equipos, deberían ser 20"
for t in equipos:
    nl, nv = (cal.home_team == t).sum(), (cal.away_team == t).sum()
    assert nl == 19 and nv == 19, f"{t}: {nl} de local y {nv} de visitante (deberían ser 19 y 19)"
assert not cal.duplicated(["home_team", "away_team"]).any(), "Hay enfrentamientos repetidos"
assert pd.to_datetime(cal.fecha).is_monotonic_increasing, "El calendario no está en orden cronológico"
print(f"Calendario {TEMPORADA}: {M} partidos, {T} equipos, validado.")

with open(PROCESSED_DIR / "elo_ratings_final.json") as f:
    elo_ini = json.load(f)
with open(PROCESSED_DIR / "team_home_form.json") as f:
    hform_ini = json.load(f)
with open(PROCESSED_DIR / "team_away_form.json") as f:
    aform_ini = json.load(f)

nuevos = [t for t in equipos if t not in elo_ini]
if nuevos:
    print(f"Sin historia (arrancan en ELO 1500): {', '.join(nuevos)}")

# elo[s, t] = ELO del equipo t en la simulación s
elo = np.tile(np.array([elo_ini.get(t, 1500.0) for t in equipos], dtype=np.float64), (N_SIMS, 1))

# Buffers circulares de los últimos 5 resultados, por sede.
# El puntero es el MISMO para todas las simulaciones: el calendario es fijo, así
# que el equipo t juega su k-ésimo partido de local en el mismo momento en todas.
hbuf = np.zeros((N_SIMS, T, WINDOW)); hn = np.zeros(T, dtype=int); hptr = np.zeros(T, dtype=int)
abuf = np.zeros((N_SIMS, T, WINDOW)); an = np.zeros(T, dtype=int); aptr = np.zeros(T, dtype=int)
for t, i in idx.items():
    for buf, n, ptr, ini in ((hbuf, hn, hptr, hform_ini), (abuf, an, aptr, aform_ini)):
        h = ini.get(t, [])[-WINDOW:]
        if h:
            buf[:, i, :len(h)] = h
            n[i] = len(h)
            ptr[i] = len(h) % WINDOW


def winrate(buf, n, i):
    """Promedio de los últimos min(n, 5) resultados. 0 si no hay historia,
    igual que calculate_recent_winrate en 02_feature_engineering.py."""
    return buf[:, i, :n[i]].mean(axis=1) if n[i] > 0 else np.zeros(buf.shape[0])


def empujar(buf, n, ptr, i, valores):
    buf[:, i, ptr[i]] = valores
    ptr[i] = (ptr[i] + 1) % WINDOW
    n[i] = min(n[i] + 1, WINDOW)


# =========================================
# 3. SIMULACIÓN
# =========================================
puntos = np.zeros((N_SIMS, T))
ganados = np.zeros((N_SIMS, T)); empatados = np.zeros((N_SIMS, T)); perdidos = np.zeros((N_SIMS, T))
prob_media = np.zeros((M, 3))        # P(A), P(D), P(H) promediadas entre simulaciones
frec = np.zeros((M, 3))              # con qué frecuencia salió cada resultado
res_todas = np.zeros((M, N_SIMS), dtype=np.int8)   # resultado de cada partido en cada simulación
n_empates = np.zeros(N_SIMS)                        # empates por simulación

print(f"\nSimulando {N_SIMS:,} temporadas ({M} llamadas al modelo, una por partido)...")
for m, row in cal.iterrows():
    ih, ia = idx[row.home_team], idx[row.away_team]
    elo_h, elo_a = elo[:, ih], elo[:, ia]

    X = np.column_stack([elo_h - elo_a, winrate(hbuf, hn, ih) - winrate(abuf, an, ia)])
    P = np.asarray(predict_proba(pd.DataFrame(X, columns=FEATURES)), dtype=np.float64)
    P = P / P.sum(axis=1, keepdims=True)
    prob_media[m] = P.mean(axis=0)

    # sorteo vectorizado: una muestra por simulación
    res = (rng.random(N_SIMS)[:, None] > P.cumsum(axis=1)).sum(axis=1)
    res = np.clip(res, 0, 2)
    es_A, es_D, es_H = res == iA, res == iD, res == iH
    frec[m] = [es_A.mean(), es_D.mean(), es_H.mean()]
    res_todas[m] = res
    n_empates += es_D

    # puntos y récord
    puntos[:, ih] += np.where(es_H, 3, np.where(es_D, 1, 0))
    puntos[:, ia] += np.where(es_A, 3, np.where(es_D, 1, 0))
    ganados[:, ih] += es_H; empatados[:, ih] += es_D; perdidos[:, ih] += es_A
    ganados[:, ia] += es_A; empatados[:, ia] += es_D; perdidos[:, ia] += es_H

    # ELO, con la misma fórmula de 02_feature_engineering.py
    s_home = np.where(es_H, 1.0, np.where(es_D, 0.5, 0.0))
    esperado_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
    elo[:, ih] = elo_h + K_ELO * (s_home - esperado_h)
    elo[:, ia] = elo_a + K_ELO * ((1.0 - s_home) - (1.0 - esperado_h))

    # forma reciente: 1 si el equipo ganó ESE partido en su sede
    empujar(hbuf, hn, hptr, ih, es_H.astype(float))
    empujar(abuf, an, aptr, ia, es_A.astype(float))

    if (m + 1) % 50 == 0:
        print(f"  {m + 1}/{M} partidos")

# =========================================
# 4. TEMPORADA REPRESENTATIVA
# =========================================
# Para ilustrar "cómo se vería" la temporada hace falta UNA simulación concreta.
# La tentación es tomar la más cercana al vector de puntos promedio, pero eso
# elige mal: promediar 10.000 temporadas aplasta la tabla (el campeón promedio
# saca menos que un campeón real y el colista más que un colista real), así que
# la simulación más parecida a ese promedio resulta anormalmente plana y
# anormalmente predecible — medido, queda en el percentil 4 de dispersión y en
# el 99 de coincidencia con el orden esperado. Sería una temporada sin sorpresas,
# que es justo lo que nunca pasa.
#
# En vez de eso se busca una temporada MEDIANA en cinco aspectos a la vez:
# cuántos puntos hizo el campeón, cuántos el colista, qué tan dispersa quedó la
# tabla, cuántos empates hubo, y qué tanto coincidió el orden final con el
# esperado. Se convierte cada uno a percentil y se elige la simulación cuyos
# cinco percentiles estén más cerca de 0.50.

def pct_rank(v):
    """Percentil de cada elemento dentro de su propio vector (0 a 1)."""
    r = np.empty(len(v), dtype=float)
    r[np.argsort(v, kind="stable")] = np.arange(1, len(v) + 1)
    return r / len(v)


def rangos(m):
    """Rango (1 = más puntos) de cada equipo, fila por fila."""
    r = np.empty_like(m, dtype=float)
    np.put_along_axis(r, np.argsort(-m, axis=1),
                      np.arange(1, m.shape[1] + 1)[None, :].repeat(m.shape[0], 0), axis=1)
    return r


puntos_medios_vec = puntos.mean(axis=0)
rank_esperado = rangos(puntos_medios_vec[None, :])[0]
rank_sims = rangos(puntos)
# correlación de Spearman = Pearson sobre los rangos
rs_c = rank_sims - rank_sims.mean(axis=1, keepdims=True)
re_c = rank_esperado - rank_esperado.mean()
corr_orden = (rs_c @ re_c) / (np.sqrt((rs_c ** 2).sum(axis=1)) * np.sqrt((re_c ** 2).sum()))

criterios = {
    "puntos del campeon": puntos.max(axis=1),
    "puntos del colista": puntos.min(axis=1),
    "dispersion de la tabla": puntos.std(axis=1),
    "empates en la temporada": n_empates,
    "coincidencia con el orden esperado": corr_orden,
}
pcts = np.column_stack([pct_rank(v) for v in criterios.values()])
i_tipica = int(np.argmin(((pcts - 0.5) ** 2).sum(axis=1)))

# Otras dos temporadas de interés, para poder contrastarlas en la app:
#   - la que más se parece al orden esperado (Arsenal, City, United...): es la
#     más "limpia" para presentar, pero precisamente por eso es POCO típica:
#     una temporada real trae sorpresas y esta casi no tiene.
#   - la más caótica, el otro extremo.
i_esperada = int(np.argmax(corr_orden))
i_caotica = int(np.argmin(corr_orden))

print(f"\nTemporada representativa elegida: simulación #{i_tipica + 1} de {N_SIMS:,}")
print(f"  {'criterio':36s} {'mediana':>9s} {'elegida':>9s} {'percentil':>10s}")
for j, (nombre, vals) in enumerate(criterios.items()):
    print(f"  {nombre:36s} {np.median(vals):9.2f} {vals[i_tipica]:9.2f} {pcts[i_tipica, j]:10.2f}")

ejemplo = [CLASSES[int(c)] for c in res_todas[:, i_tipica]]
puntos_ejemplo = puntos[i_tipica]

# =========================================
# 5. TABLA: posiciones y probabilidades
# =========================================
# Desempate por puntos con un ruido minúsculo, para que los empates a puntos
# no se resuelvan siempre a favor del mismo equipo por su orden alfabético.
orden = (-(puntos + rng.random(puntos.shape) * 1e-6)).argsort(axis=1)
posicion = np.empty_like(orden)
np.put_along_axis(posicion, orden, np.arange(1, T + 1)[None, :].repeat(N_SIMS, 0), axis=1)

tabla = pd.DataFrame({
    "equipo": equipos,
    "puntos_medios": puntos.mean(axis=0).round(1),
    "puntos_p5": np.percentile(puntos, 5, axis=0).round(0),
    "puntos_p95": np.percentile(puntos, 95, axis=0).round(0),
    "posicion_media": posicion.mean(axis=0).round(1),
    "prob_campeon": (posicion == 1).mean(axis=0).round(4),
    "prob_top4": (posicion <= 4).mean(axis=0).round(4),
    "prob_descenso": (posicion >= 18).mean(axis=0).round(4),
    "ganados_medios": ganados.mean(axis=0).round(1),
    "empatados_medios": empatados.mean(axis=0).round(1),
    "perdidos_medios": perdidos.mean(axis=0).round(1),
    "elo_final_medio": elo.mean(axis=0).round(1),
    "puntos_ejemplo": puntos_ejemplo.round(0),
}).sort_values("puntos_medios", ascending=False).reset_index(drop=True)
tabla.insert(0, "puesto_esperado", range(1, T + 1))

# =========================================
# 6. PARTIDO A PARTIDO
# =========================================
partidos = cal.copy()
partidos["prob_local"] = prob_media[:, iH].round(4)
partidos["prob_empate"] = prob_media[:, iD].round(4)
partidos["prob_visitante"] = prob_media[:, iA].round(4)
partidos["resultado_mas_probable"] = np.array(["A", "D", "H"])[
    np.argmax(prob_media[:, [iA, iD, iH]], axis=1)]
partidos["frec_local"] = frec[:, iH].round(4)
partidos["frec_empate"] = frec[:, iD].round(4)
partidos["frec_visitante"] = frec[:, iA].round(4)
partidos["ejemplo_simulado"] = ejemplo   # la temporada representativa elegida arriba

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
partidos.to_csv(PROCESSED_DIR / f"prediccion_partidos_{TEMPORADA.replace('-', '_')}.csv", index=False)
tabla.to_csv(PROCESSED_DIR / f"prediccion_tabla_{TEMPORADA.replace('-', '_')}.csv", index=False)

# Todas las simulaciones, para que la app pueda mostrar cualquiera de ellas.
# Comprimido pesa ~1 MB; en crudo serían 4 MB.
np.savez_compressed(
    PROCESSED_DIR / f"simulaciones_{TEMPORADA.replace('-', '_')}.npz",
    resultados=res_todas.astype(np.int8),          # (partido, simulacion) -> 0=A, 1=D, 2=H
    puntos=puntos.astype(np.int16),                # (simulacion, equipo)
    equipos=np.array(equipos),
    clases=np.array(CLASSES),
    destacadas=np.array([i_tipica, i_esperada, i_caotica]),
    etiquetas=np.array(["representativa", "mas_parecida_a_lo_esperado", "mas_caotica"]),
)

# =========================================
# 7. RESUMEN EN PANTALLA
# =========================================
print(f"\n{'=' * 78}\nTABLA ESPERADA {TEMPORADA}  ({N_SIMS:,} simulaciones)\n{'=' * 78}")
print(tabla[["puesto_esperado", "equipo", "puntos_medios", "puntos_p5", "puntos_p95",
             "prob_campeon", "prob_top4", "prob_descenso"]].to_string(index=False))

tot = puntos.sum(axis=1)
emp = empatados.sum(axis=1) / 2
print(f"\nControl de coherencia (una temporada real reparte ~1035 puntos con ~100 empates):")
print(f"  puntos repartidos : {tot.mean():.1f}  (rango {tot.min():.0f} a {tot.max():.0f})")
print(f"  empates por temporada: {emp.mean():.1f}  ({emp.mean() / M * 100:.1f}% de los partidos)")
campeon = puntos.max(axis=1)
colista = puntos.min(axis=1)
print(f"  puntos del campeon   : {campeon.mean():.1f}  (en 2025-2026 el campeon real hizo 84)")
print(f"  puntos del colista   : {colista.mean():.1f}")
print(f"\nGuardado en {PROCESSED_DIR}/:")
print(f"  prediccion_partidos_{TEMPORADA.replace('-', '_')}.csv  ({M} partidos)")
print(f"  prediccion_tabla_{TEMPORADA.replace('-', '_')}.csv     ({T} equipos)")
print(f"  simulaciones_{TEMPORADA.replace('-', '_')}.npz          (las {N_SIMS:,} temporadas completas)")
print(f"\nTemporadas destacadas: representativa #{i_tipica + 1} | "
      f"mas parecida a lo esperado #{i_esperada + 1} (corr {corr_orden[i_esperada]:.2f}) | "
      f"mas caotica #{i_caotica + 1} (corr {corr_orden[i_caotica]:.2f})")
print("\n>>> Siguiente paso: streamlit run app/app.py  (pestaña 'Temporada 2026-2027')")
