"""
02_feature_engineering.py
Recicla las funciones de ELO y win_rate_5 del notebook original y las aplica
sobre la base unificada completa (2006-07 -> 2025-26), respetando el orden
cronológico partido a partido.

Además de las features, guarda el ESTADO FINAL de:
  - ELO de cada equipo               -> elo_ratings_final.json
  - Últimos resultados como LOCAL    -> team_home_form.json
  - Últimos resultados como VISITANTE -> team_away_form.json
Estos 3 archivos son los que la app de Streamlit usa para calcular las
features de un partido hipotético entre dos equipos "hoy".

Ejecutar desde cualquier lado con: python src/02_feature_engineering.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(PROCESSED_DIR / "dfpremier_unificado.csv")

# =========================================
# FUNCIONES ELO (idénticas al notebook original, K=30)
# =========================================
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, score_a, k=30):
    exp_a = expected_score(elo_a, elo_b)
    exp_b = 1 - exp_a
    new_a = elo_a + k * (score_a - exp_a)
    new_b = elo_b + k * ((1 - score_a) - exp_b)
    return new_a, new_b

def calculate_elo(df, k=30, initial_elo=1500):
    elo_ratings = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home, away, result = row["home_team"], row["away_team"], row["result"]

        if home not in elo_ratings:
            elo_ratings[home] = initial_elo
        if away not in elo_ratings:
            elo_ratings[away] = initial_elo

        home_elo, away_elo = elo_ratings[home], elo_ratings[away]
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        if result == "H":
            score_home = 1
        elif result == "A":
            score_home = 0
        else:
            score_home = 0.5

        new_home, new_away = update_elo(home_elo, away_elo, score_home, k=k)
        elo_ratings[home], elo_ratings[away] = new_home, new_away

    df["home_elo"], df["away_elo"] = home_elos, away_elos
    return df, elo_ratings

# =========================================
# WIN RATE ÚLTIMOS 5 (idéntica al notebook original)
# home_winrate_5  = de los últimos 5 partidos JUGANDO DE LOCAL, % ganados
# away_winrate_5  = de los últimos 5 partidos JUGANDO DE VISITANTE, % ganados
# =========================================
def calculate_recent_winrate(df, team_col, result_col, window=5):
    winrates = []
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    team_history = {team: [] for team in all_teams}

    for _, row in df.iterrows():
        team, result = row[team_col], row[result_col]
        hist = team_history[team][-window:]
        winrate = np.mean(hist) if hist else 0
        winrates.append(winrate)

        if team_col == "home_team":
            win = 1 if result == "H" else 0
        elif team_col == "away_team":
            win = 1 if result == "A" else 0
        else:
            win = 0
        team_history[team].append(win)

    return winrates, team_history

# =========================================
# CONSTRUCCIÓN DE FEATURES
# =========================================
df, final_elos = calculate_elo(df, k=30, initial_elo=1500)
df["home_winrate_5"], home_history_final = calculate_recent_winrate(df, "home_team", "result", 5)
df["away_winrate_5"], away_history_final = calculate_recent_winrate(df, "away_team", "result", 5)
df["elo_diff"] = df["home_elo"] - df["away_elo"]
df["winrate_diff"] = df["home_winrate_5"] - df["away_winrate_5"]

print("Shape con features:", df.shape)
print(df[["home_team", "away_team", "result", "home_elo", "away_elo",
          "home_winrate_5", "away_winrate_5", "elo_diff", "winrate_diff", "season"]].tail(10))

# =========================================
# GUARDAR TODO
# =========================================
df.to_csv(PROCESSED_DIR / "dfpremier_features.csv", index=False)
print("\nGuardado dfpremier_features.csv")

with open(PROCESSED_DIR / "elo_ratings_final.json", "w") as f:
    json.dump(final_elos, f, indent=2)
print("Guardado elo_ratings_final.json (ELO de cada equipo al cierre de 2025-2026)")

# Solo guardamos los últimos 5 resultados de cada equipo (lo que realmente
# necesita la app para calcular winrate_5 "hoy")
home_form_last5 = {team: hist[-5:] for team, hist in home_history_final.items()}
away_form_last5 = {team: hist[-5:] for team, hist in away_history_final.items()}

with open(PROCESSED_DIR / "team_home_form.json", "w") as f:
    json.dump(home_form_last5, f, indent=2)
with open(PROCESSED_DIR / "team_away_form.json", "w") as f:
    json.dump(away_form_last5, f, indent=2)
print("Guardado team_home_form.json y team_away_form.json (últimos 5 resultados por equipo)")
