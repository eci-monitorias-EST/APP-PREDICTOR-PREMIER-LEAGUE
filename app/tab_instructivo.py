"""
tab_instructivo.py
Pestaña "¿Cómo se usa?": guía paso a paso para usar la app.
"""
import streamlit as st


def _miles(n: int) -> str:
    """7600 -> '7.600' (separador de miles en español)."""
    return f"{int(n):,}".replace(",", ".")


def _camiseta(numero: int) -> str:
    """Marcador de paso con forma de camiseta y el número del paso."""
    return (
        '<svg class="paso-camiseta" viewBox="0 0 48 48" aria-hidden="true">'
        '<path d="M17 5 L8 9 L2 17 L8.5 22.5 L12 20 L12 44 L36 44 L36 20 '
        'L39.5 22.5 L46 17 L40 9 L31 5 C29 9 19 9 17 5 Z"/>'
        f'<text x="24" y="36" text-anchor="middle">{numero}</text>'
        "</svg>"
    )


def _pasos(pasos: list) -> str:
    """Lista ordenada de pasos a partir de pares (titulo, texto)."""
    filas = "".join(
        f'<li class="paso">{_camiseta(i)}'
        f'<div><div class="paso-titulo">{titulo}</div>'
        f'<div class="paso-texto">{texto}</div></div></li>'
        for i, (titulo, texto) in enumerate(pasos, start=1)
    )
    return f'<ol class="pasos">{filas}</ol>'


def render_instructivo(n_partidos: int, n_simulaciones: int, local: str,
                       visitante: str, probs: dict, mostrar_barra) -> None:
    """Dibuja la pestaña completa."""
    sims = _miles(n_simulaciones)

    st.markdown(
        '<div class="guia-intro">'
        '<div class="guia-titulo">Ingeniero estadístico por un día</div>'
        '<div class="guia-texto">Un ingeniero estadístico usa datos del pasado para '
        'calcular qué tan probable es que algo pase. Hoy vas a hacer ese trabajo con '
        'la Premier League, la liga de fútbol de Inglaterra.</div>'
        f'<div class="guia-texto">La app aprendió de {_miles(n_partidos)} partidos '
        'jugados desde 2006. Con eso puede decirte qué tan probable es que un equipo '
        f'le gane a otro, y hasta jugar la próxima temporada completa {sims} veces.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------
    # PREDECIR UN PARTIDO
    # -------------------------------------------------
    st.subheader("Predecir un partido")
    st.markdown(_pasos([
        ("Abre la pestaña <i>Predecir un partido</i>",
         "Está arriba, en la barra de pestañas."),
        ("Elige los dos equipos",
         "En <b>Equipo local</b> va el que juega en su estadio y en <b>Equipo "
         "visitante</b> el que llega de visita. Tienen que ser distintos."),
        ("Presiona <i>Predecir resultado</i>",
         "La app calcula tres probabilidades: que gane el local, que empaten o que "
         "gane el visitante."),
        ("Lee la barra",
         "Se ve como el ejemplo de aquí abajo."),
    ]), unsafe_allow_html=True)

    st.markdown(
        f'<div class="ejemplo-etiqueta">Ejemplo: {local} de local contra {visitante}</div>',
        unsafe_allow_html=True,
    )
    mostrar_barra(local, visitante, probs)

    p_local = round(probs["home_win"] * 100)
    p_empate = round(probs["draw"] * 100)
    p_visit = round(probs["away_win"] * 100)
    st.markdown(
        '<div class="nota-clave">'
        '<div class="guia-texto">Cada número dice qué tan probable es ese resultado, y '
        'los tres suman 100%. Entre más ancha la franja de un color, más probable es.</div>'
        '<div class="nota-clave-titulo">Una probabilidad no es una promesa</div>'
        f'<div class="guia-texto">Si este partido se jugara 100 veces, {local} ganaría '
        f'unas {p_local}, habría unos {p_empate} empates y {visitante} ganaría unas '
        f'{p_visit}. Por eso a veces gana el que tenía menos opciones: eso también '
        'estaba dentro de lo posible.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="guia-texto guia-texto--suave">Debajo de la barra aparece '
        '<b>Ver detalle</b>. Ahí están los dos datos que usó la app para calcular: el '
        'ELO y la forma reciente de cada equipo. Los explicamos en la pestaña '
        '<i>Diccionario de datos</i>.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # -------------------------------------------------
    # TEMPORADA SIMULADA
    # -------------------------------------------------
    st.subheader("Temporada 2026-2027")
    st.markdown(_pasos([
        ("Abre la pestaña <i>Temporada 2026-2027</i>",
         f"La app jugó la próxima temporada, con sus 380 partidos, {sims} veces. "
         "En cada partido el resultado se sortea como si se lanzara un dado cargado: "
         "el resultado más probable sale más seguido, pero no siempre."),
        ("Mira el <i>Resumen de las simulaciones</i>",
         "Ahí está el favorito al título, el equipo con más riesgo de bajar a segunda "
         "división y la tabla de posiciones promedio. Más abajo puedes escoger un "
         "equipo y ver en qué puesto terminó en todas las temporadas."),
        ("Entra a <i>Explorar una temporada</i>",
         f"Escoge una sola de las {sims} temporadas y mira cómo terminó. Prueba con "
         "<b>La más caótica</b> para ver una llena de sorpresas."),
        ("Revisa <i>Partido a partido</i>",
         "Busca cualquier partido de la temporada y mira sus tres probabilidades. "
         "Puedes filtrar para ver solo los de tu equipo favorito."),
    ]), unsafe_allow_html=True)

    st.divider()

    # -------------------------------------------------
    # RETOS
    # -------------------------------------------------
    st.subheader("Retos")
    st.markdown(
        '<div class="guia-texto">Ya sabes usar la app. Ahora piensa como estadístico.</div>'
        '<div class="retos">'
        '<div class="reto"><div class="reto-titulo">El partido más parejo</div>'
        '<div class="reto-texto">Busca dos equipos que tengan casi la misma probabilidad '
        'de ganar. ¿Qué le pasa a la probabilidad de empate en ese partido?</div></div>'
        '<div class="reto"><div class="reto-titulo">La ventaja de jugar en casa</div>'
        '<div class="reto-texto">Pon a tu equipo favorito de local contra un rival y luego '
        'cámbialos de lado. ¿Cambian los números? ¿Por qué crees que pasa?</div></div>'
        '<div class="reto"><div class="reto-titulo">La gran sorpresa</div>'
        '<div class="reto-texto">En el <i>Resumen de las simulaciones</i>, busca la gráfica '
        '<b>¿En qué puesto termina cada equipo?</b> y escoge al equipo que va de último en '
        f'la tabla. Lee el texto de abajo: ¿cuál fue su mejor puesto en las {sims} '
        'temporadas?</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
