"""
team_colors.py
Color principal de camiseta/escudo por equipo, para pintar la barra de
probabilidades estilo Google. Si un equipo no está en el diccionario
(por ejemplo un ascendido nuevo que aún no le has puesto color), se usa
DEFAULT_COLOR.

Ajusta estos valores libremente, son aproximaciones.
"""

DEFAULT_COLOR = "#9E9E9E"  # gris neutro para equipos sin color asignado

TEAM_COLORS = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#670E36",
    "AFC Bournemouth": "#DA291C",
    "Brentford": "#E30613",
    "Brighton and Hove Albion": "#0057B8",
    "Burnley": "#6C1D45",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Ipswich Town": "#0044A9",
    "Leeds": "#FFCD00",  # antes "Leeds United": no coincidía con el dataset y caía en DEFAULT_COLOR
    "Leicester City": "#003090",
    "Liverpool": "#C8102E",
    "Luton": "#F78F1E",  # antes "Luton Town": mismo caso que Leeds
    "Manchester City": "#6CABDD",
    "Manchester United": "#DA291C",
    "Newcastle United": "#241F20",
    "Norwich City": "#FFF200",
    "Nottingham Forest": "#DD0000",
    "Sheffield United": "#EE2737",
    "Southampton": "#D71920",
    "Sunderland": "#EB172B",
    "Tottenham Hotspur": "#132257",
    "Watford": "#FBEE23",
    "West Bromwich Albion": "#122F67",
    "West Ham United": "#7A263A",
    "Wolverhampton Wanderers": "#FDB913",
    "Cardiff City": "#0070B5",
    "Huddersfield Town": "#0E63AD",
    "Hull City": "#F18A00",
    "Stoke City": "#E03A3E",
    "Swansea City": "#000000",
    "Blackburn Rovers": "#009EE0",
    "Bolton Wanderers": "#002B5C",
    "Wigan Athletic": "#1D59AF",
    "Reading": "#004494",
    "Derby County": "#000000",
    "Portsmouth": "#001489",
    "Blackpool": "#FF7900",
    "Charlton Athletic": "#D2122E",
    "Birmingham City": "#0000FF",
    "Middlesbrough": "#DA1710",
    "Queens Park Rangers": "#1D5BA4",
}


def get_team_color(team_name: str) -> str:
    """Devuelve el color del equipo, o DEFAULT_COLOR si no está registrado."""
    return TEAM_COLORS.get(team_name, DEFAULT_COLOR)
