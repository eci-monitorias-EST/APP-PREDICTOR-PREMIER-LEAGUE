"""
team_badges.py
Iniciales (3 letras) por equipo, para dibujar una "ficha" circular en el
color del equipo (ver team_colors.py) en vez de un escudo real.

Por qué no usamos los escudos oficiales: son marcas registradas de cada
club / de la Premier League, así que evitamos reproducirlos o descargarlos
desde la web. La ficha con iniciales + color cumple la misma función visual
(identificar al equipo de un vistazo) sin ese problema — es el mismo
recurso que usan paneles de datos deportivos tipo FotMob o SofaScore.

Si en algún momento consiguen sus propios assets con licencia (por ejemplo
dibujados por ustedes), pueden reemplazar render_team_badge() para que
devuelva un <img src="..."> en vez del <div> con iniciales, sin tener que
tocar app.py.
"""

# Debe usar exactamente los mismos nombres de equipo que
# data/processed/elo_ratings_final.json (y por lo tanto TEAM_COLORS).
TEAM_CODES = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "AFC Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton and Hove Albion": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Ipswich Town": "IPS",
    "Leeds": "LEE",
    "Leicester City": "LEI",
    "Liverpool": "LIV",
    "Luton": "LUT",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Norwich City": "NOR",
    "Nottingham Forest": "NFO",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Sunderland": "SUN",
    "Tottenham Hotspur": "TOT",
    "Watford": "WAT",
    "West Bromwich Albion": "WBA",
    "West Ham United": "WHU",
    "Wolverhampton Wanderers": "WOL",
    "Cardiff City": "CAR",
    "Huddersfield Town": "HUD",
    "Hull City": "HUL",
    "Stoke City": "STK",
    "Swansea City": "SWA",
    "Blackburn Rovers": "BLB",
    "Bolton Wanderers": "BOL",
    "Wigan Athletic": "WIG",
    "Reading": "REA",
    "Derby County": "DER",
    "Portsmouth": "POR",
    "Blackpool": "BLP",
    "Charlton Athletic": "CHA",
    "Birmingham City": "BIR",
    "Middlesbrough": "MID",
    "Queens Park Rangers": "QPR",
}


def get_team_code(team_name: str) -> str:
    """Iniciales para la ficha. Si el equipo no está en el diccionario
    (por ejemplo un ascendido nuevo), usa las primeras 3 letras del
    nombre como respaldo en vez de fallar."""
    return TEAM_CODES.get(team_name, team_name[:3].upper())


def render_team_badge(team_name: str, color: str) -> str:
    """HTML de una ficha circular con las iniciales del equipo, lista para
    insertar dentro de un st.markdown(..., unsafe_allow_html=True)."""
    code = get_team_code(team_name)
    return f'<div class="team-badge" style="background-color:{color};">{code}</div>'
