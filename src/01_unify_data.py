"""
01_unify_data.py
Unifica data/raw/dfpremier_actualizado.csv (2006-07 -> 2023-24) con
data/raw/202420252026.csv (2024/25 y 2025/26) en un único dataframe con
el esquema histórico: home_team, away_team, home_goals, away_goals, result, season

Ejecutar desde cualquier lado con: python src/01_unify_data.py
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =========================================
# 1. CARGA
# =========================================
dfh = pd.read_csv(RAW_DIR / "dfpremier_actualizado.csv")
dfnew = pd.read_csv(RAW_DIR / "202420252026.csv")

print("Histórico:", dfh.shape, "| Nuevo:", dfnew.shape)

# =========================================
# 2. RENAME DICT EXTENDIDO
#    (el original del notebook + las variantes que trae la fuente nueva)
# =========================================
rename_dict = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton and Hove Albion",
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Leicester": "Leicester City",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Ipswich": "Ipswich Town",
}

# =========================================
# 3. LIMPIEZA DEL ARCHIVO NUEVO -> ESQUEMA HISTÓRICO
# =========================================
dfnew_clean = dfnew.rename(columns={
    "homeTeam": "home_team",
    "awayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
})

dfnew_clean["matchDate_parsed"] = pd.to_datetime(
    dfnew_clean["matchDate"], format="%d-%m-%y %H:%M"
)
dfnew_clean = dfnew_clean.sort_values(["Season", "matchDate_parsed"]).reset_index(drop=True)
dfnew_clean["season"] = dfnew_clean["Season"].str.replace("/", "-", regex=False)
dfnew_clean = dfnew_clean[["home_team", "away_team", "home_goals", "away_goals", "result", "season"]]

# =========================================
# 4. APLICAR RENAME A AMBAS BASES
# =========================================
dfh["home_team"] = dfh["home_team"].replace(rename_dict)
dfh["away_team"] = dfh["away_team"].replace(rename_dict)
dfnew_clean["home_team"] = dfnew_clean["home_team"].replace(rename_dict)
dfnew_clean["away_team"] = dfnew_clean["away_team"].replace(rename_dict)

# =========================================
# 5. CONCATENAR RESPETANDO ORDEN CRONOLÓGICO
# =========================================
df_unificado = pd.concat([dfh, dfnew_clean], ignore_index=True)

# =========================================
# 6. VALIDACIONES
# =========================================
print("\n=== VALIDACIÓN ===")
print("Shape final:", df_unificado.shape)
print("Partidos por temporada:")
print(df_unificado["season"].value_counts().sort_index())
print("\nNulos por columna:")
print(df_unificado.isnull().sum())
print("\nValores únicos de 'result':", df_unificado["result"].unique())
print("Equipos únicos:", df_unificado["home_team"].nunique())

# =========================================
# 7. GUARDAR
# =========================================
out_path = PROCESSED_DIR / "dfpremier_unificado.csv"
df_unificado.to_csv(out_path, index=False)
print(f"\nGuardado en {out_path}")
