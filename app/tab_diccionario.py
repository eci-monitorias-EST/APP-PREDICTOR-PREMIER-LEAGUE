"""
tab_diccionario.py
Pestaña "Diccionario de datos": variables del proyecto presentadas como láminas.
"""
import numpy as np
import pandas as pd
import streamlit as st

TIPOS = {
    "Texto": ("#9DB8FF", "palabras o nombres"),
    "Categoría": ("#F2A0A0", "una opción de una lista corta"),
    "Número entero": ("#7BD389", "sin decimales"),
    "Número decimal": ("#5ED0E6", "puede tener decimales"),
    "Porcentaje": ("#FFC94D", "de 0% a 100%"),
}

PARTIDO_REAL = {"local": "Liverpool", "visitante": "AFC Bournemouth",
                "goles_local": 4, "goles_visitante": 2, "resultado": "H"}


def _miles(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def _lamina(num: int, e: dict) -> str:
    color = TIPOS[e["tipo"]][0]
    codigos = "".join(f"<code>{c}</code>" for c in e["codigo"])
    return (
        '<div class="lamina">'
        f'<div class="lamina-cara" style="background-color:{color};">'
        f'<div class="lamina-top"><span class="lamina-num">{num}</span>'
        f'<span class="lamina-tipo">{e["tipo"]}</span></div>'
        f'<div class="lamina-valor">{e["cara"]}</div>'
        f'<div class="lamina-pie">{e["pie"]}</div>'
        '</div>'
        '<div class="lamina-cuerpo">'
        f'<div class="lamina-nombre">{e["nombre"]}</div>'
        f'<div class="lamina-codigo">{codigos}</div>'
        f'<div class="lamina-texto">{e["que_es"]}</div>'
        f'<div class="lamina-valores"><b>Valores posibles:</b> {e["valores"]}</div>'
        '</div>'
        '</div>'
    )


def _entradas(elo_ratings, home_form, away_form, tabla_sim, local, visitante,
              probs, n_simulaciones):
    """Arma los tres grupos de láminas con ejemplos tomados de los datos."""
    sims = _miles(n_simulaciones)
    pr = PARTIDO_REAL

    elo_local = elo_ratings.get(local, 1500)
    elo_visit = elo_ratings.get(visitante, 1500)
    eq_min = min(elo_ratings, key=elo_ratings.get)
    eq_max = max(elo_ratings, key=elo_ratings.get)

    forma = home_form.get(local, [])
    forma_visit = away_form.get(visitante, [])
    ganados, jugados = int(sum(forma)), len(forma)
    forma_pct = round(np.mean(forma) * 100) if forma else 0
    wr_diff = (np.mean(forma) if forma else 0) - (np.mean(forma_visit) if forma_visit else 0)

    if tabla_sim is not None and local in set(tabla_sim["equipo"]):
        fila = tabla_sim.set_index("equipo").loc[local]
        eq_tabla = local
    elif tabla_sim is not None:
        fila = tabla_sim.iloc[0]
        eq_tabla = fila["equipo"]
    else:
        fila, eq_tabla = None, local
    puntos_cara = f"{fila['puntos_medios']:.1f}" if fila is not None else "–"
    campeon_cara = f"{fila['prob_campeon'] * 100:.0f}%" if fila is not None else "–"

    cancha = [
        {"nombre": "Equipos", "codigo": ["home_team", "away_team"], "tipo": "Texto",
         "cara": pr["local"], "pie": f"local contra {pr['visitante']}",
         "que_es": "El nombre del equipo que juega en su estadio (local) y el del "
                   "que llega de visita (visitante).",
         "valores": f"cualquiera de los {len(elo_ratings)} equipos que han jugado la "
                    "Premier desde 2006."},
        {"nombre": "Goles", "codigo": ["home_goals", "away_goals"],
         "tipo": "Número entero",
         "cara": f"{pr['goles_local']}-{pr['goles_visitante']}",
         "pie": f"{pr['local']} {pr['goles_local']}, {pr['visitante']} {pr['goles_visitante']}",
         "que_es": "Cuántos goles hizo cada equipo en el partido.",
         "valores": "0, 1, 2, 3... nunca negativos. El máximo en la base es 9."},
        {"nombre": "Resultado", "codigo": ["result"], "tipo": "Categoría",
         "cara": pr["resultado"], "pie": "ganó el local",
         "que_es": "Quién ganó el partido, escrito con una sola letra.",
         "valores": "H si gana el local, D si empatan y A si gana el visitante. "
                    "Vienen del inglés: <i>home</i>, <i>draw</i>, <i>away</i>."},
    ]

    calculos = [
        {"nombre": "ELO", "codigo": ["home_elo", "away_elo"], "tipo": "Número decimal",
         "cara": f"{elo_local:.0f}", "pie": f"ELO de {local}",
         "que_es": "Un puntaje de fuerza, como el ranking de un videojuego. Ganar lo "
                   "sube y perder lo baja, y ganarle a un rival fuerte da más puntos "
                   "que ganarle a uno débil.",
         "valores": f"todos empiezan en 1500. Hoy van de {elo_ratings[eq_min]:.0f} "
                    f"({eq_min}) a {elo_ratings[eq_max]:.0f} ({eq_max})."},
        {"nombre": "Forma reciente", "codigo": ["home_winrate_5", "away_winrate_5"],
         "tipo": "Porcentaje",
         "cara": f"{forma_pct}%",
         "pie": (f"{local} ganó {ganados} de sus últimos {jugados} en casa" if jugados
                 else f"{local} no tiene partidos recientes en casa"),
         "que_es": "Qué porcentaje de sus últimos 5 partidos ganó el equipo. Para el "
                   "local se cuentan solo sus partidos en casa y para el visitante, "
                   "solo los de visita.",
         "valores": "0%, 20%, 40%, 60%, 80% o 100%."},
        {"nombre": "Ventaja del local", "codigo": ["elo_diff", "winrate_diff"],
         "tipo": "Número decimal",
         "cara": f"{elo_local - elo_visit:+.0f}",
         "pie": f"{local} contra {visitante}, en ELO",
         "que_es": "El ELO del local menos el del visitante, y lo mismo con la forma "
                   f"(en el ejemplo da {wr_diff:+.1f}). Si es positivo, el local llega "
                   "mejor. <b>Son los únicos dos datos que mira el modelo.</b>",
         "valores": "en ELO casi siempre entre -500 y 500; en forma, entre -1 y 1."},
    ]

    prediccion = [
        {"nombre": "Probabilidad del resultado",
         "codigo": ["prob_local", "prob_empate", "prob_visitante"], "tipo": "Porcentaje",
         "cara": f"{probs['home_win'] * 100:.0f}%",
         "pie": f"gana {local} de local contra {visitante}",
         "que_es": "Qué tan probable es cada resultado según el modelo. Las tres "
                   "probabilidades de un partido suman 100%.",
         "valores": "de 0% a 100%."},
        {"nombre": "Puntos esperados", "codigo": ["puntos_medios"],
         "tipo": "Número decimal",
         "cara": puntos_cara, "pie": f"{eq_tabla} en 2026-2027",
         "que_es": f"Los puntos que hace el equipo en promedio en las {sims} temporadas "
                   "simuladas. Ganar da 3 puntos, empatar 1 y perder 0.",
         "valores": "de 0 a 114, porque son 38 partidos de máximo 3 puntos cada uno."},
        {"nombre": "Campeón, top 4 o descenso",
         "codigo": ["prob_campeon", "prob_top4", "prob_descenso"], "tipo": "Porcentaje",
         "cara": campeon_cara, "pie": f"{eq_tabla} sale campeón",
         "que_es": f"En qué porcentaje de las {sims} temporadas simuladas el equipo quedó "
                   "campeón, entre los 4 primeros o en los 3 últimos puestos, que bajan "
                   "de categoría.",
         "valores": "de 0% a 100%."},
    ]

    return [
        ("Lo que pasó en la cancha",
         f"Datos reales de cada partido. El ejemplo es {pr['local']} "
         f"{pr['goles_local']}-{pr['goles_visitante']} {pr['visitante']}, el primer "
         "partido de la temporada 2025-2026.", cancha),
        ("Lo que calculamos con esos datos",
         "Números que resumen qué tan fuerte llega cada equipo a un partido.", calculos),
        ("Lo que predice la app",
         f"Lo que sale del modelo y de las {sims} simulaciones.", prediccion),
    ]


TABLA_TECNICA = pd.DataFrame(
    [
        ["home_team, away_team", "Texto (str)", "44 nombres de equipo",
         "Nombre del equipo local y del visitante."],
        ["home_goals, away_goals", "Entero (int)", "0 a 9",
         "Goles de cada equipo al final del partido."],
        ["result", "Categoría (str)", "H, D, A",
         "Resultado final: gana local, empate o gana visitante."],
        ["home_elo, away_elo", "Decimal (float)", "Inicia en 1500",
         "Rating ELO antes del partido (K = 30, acumulado desde 2006-2007)."],
        ["home_winrate_5, away_winrate_5", "Decimal (float)", "0, 0.2, 0.4, 0.6, 0.8, 1",
         "Proporción de victorias en los últimos 5 partidos de local (o de visitante)."],
        ["elo_diff, winrate_diff", "Decimal (float)", "-534 a 524 / -1 a 1",
         "Local menos visitante. Son las únicas entradas del modelo."],
        ["prob_local, prob_empate, prob_visitante", "Decimal (float)", "0 a 1, suman 1",
         "Probabilidad de cada resultado según el modelo."],
        ["puntos_medios", "Decimal (float)", "0 a 114",
         "Puntos promedio del equipo entre todas las simulaciones."],
        ["prob_campeon, prob_top4, prob_descenso", "Decimal (float)", "0 a 1",
         "Fracción de simulaciones en que el equipo termina 1.º, entre los 4 primeros "
         "o en los 3 últimos."],
    ],
    columns=["Variable", "Tipo de dato", "Valores", "Descripción"],
    index=pd.RangeIndex(1, 10, name="#"),
)


def render_diccionario(elo_ratings, home_form, away_form, tabla_sim, local, visitante,
                       probs, n_simulaciones) -> None:
    st.markdown(
        '<div class="guia-intro">'
        '<div class="guia-titulo">Diccionario de datos</div>'
        '<div class="guia-texto">Es la lista de los datos que usa un proyecto: qué '
        'significa cada uno, de qué tipo es y qué valores puede tomar.</div>'
        '<div class="guia-texto">Aquí cada dato es una lámina. Están en el orden en que '
        'viajan por el proyecto: primero lo que pasó en la cancha, después lo que '
        'calculamos con eso y al final lo que predice la app.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    leyenda = "".join(
        f'<div class="tipo-item"><span class="tipo-muestra" style="background-color:{c};">'
        f'</span><b>{nombre}</b>&nbsp;{ayuda}</div>'
        for nombre, (c, ayuda) in TIPOS.items()
    )
    st.markdown(
        '<div class="tipos">'
        '<div class="tipos-titulo">El color de la lámina dice el tipo de dato</div>'
        f'<div class="tipos-lista">{leyenda}</div></div>',
        unsafe_allow_html=True,
    )

    num = 1
    for titulo, intro, entradas in _entradas(elo_ratings, home_form, away_form, tabla_sim,
                                             local, visitante, probs, n_simulaciones):
        st.subheader(titulo)
        st.markdown(f'<div class="guia-texto guia-texto--suave">{intro}</div>',
                    unsafe_allow_html=True)
        laminas = "".join(_lamina(num + i, e) for i, e in enumerate(entradas))
        st.markdown(f'<div class="album">{laminas}</div>', unsafe_allow_html=True)
        num += len(entradas)

    st.divider()
    with st.expander("Ver la versión de ingeniero (tabla)"):
        st.caption("La misma información en el formato que se usa en proyectos reales. "
                   "Los porcentajes se guardan como decimales entre 0 y 1.")
        st.table(TABLA_TECNICA)
