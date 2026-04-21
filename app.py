import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
from google_sheets_client import get_gspread_client, get_division_data, get_available_birth_years, get_tarjetas_data

# ---------- Funciones auxiliares ----------

def parse_resultado(resultado):
    """
    Extrae los puntos del partido y los puntos para la tabla.
    Identifica casos "WO" o "GP" (Walkover / Gana Puntos).
    Ej: "25 [4]" → (25, 4)
    """
    res_str = str(resultado).strip().upper()
    if res_str in ["-", "", "PENDIENTE"]:
        return None, None
    
    # Manejo de Walkover / Puntos cedidos
    if "W.O." in res_str or "WO" in res_str or "GP" in res_str:
        return 28, 5  # 28-0 y 5 puntos bonus oficial
    if "PP" in res_str:
        return 0, 0
        
    match = re.match(r"(\d+)\s*\[(\d+)\]", res_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        # Fallback si solo cargaron los tantos
        num_match = re.search(r"(\d+)", res_str)
        if num_match:
            return int(num_match.group(1)), 0
        return None, None

def procesar_partidos(df):
    posiciones = {}

    for _, row in df.iterrows():
        local = row["Local"]
        visitante = row["Visitante"]
        res_local, pts_local = parse_resultado(row["ResultadoL"])
        res_visitante, pts_visitante = parse_resultado(row["ResultadoV"])

        if None in (res_local, res_visitante):
            continue

        for equipo in [local, visitante]:
            if equipo not in posiciones:
                posiciones[equipo] = {
                    "Equipo": equipo,
                    "PTS": 0,
                    "PJ": 0,
                    "PG": 0,
                    "PE": 0,
                    "PP": 0,
                    "PB": 0,
                    "PF": 0,
                    "PC": 0,
                    "DIF": 0
                }

        posiciones[local]["PTS"] += pts_local
        posiciones[visitante]["PTS"] += pts_visitante

        posiciones[local]["PJ"] += 1
        posiciones[visitante]["PJ"] += 1

        posiciones[local]["PF"] += res_local
        posiciones[local]["PC"] += res_visitante
        posiciones[visitante]["PF"] += res_visitante
        posiciones[visitante]["PC"] += res_local

        if res_local > res_visitante:
            posiciones[local]["PG"] += 1
            posiciones[visitante]["PP"] += 1
        elif res_local < res_visitante:
            posiciones[visitante]["PG"] += 1
            posiciones[local]["PP"] += 1
        else:
            posiciones[local]["PE"] += 1
            posiciones[visitante]["PE"] += 1



    # Calcular diferencia y Puntos Bonus calculados matemáticamente
    for equipo in posiciones:
        pos = posiciones[equipo]
        pos["DIF"] = pos["PF"] - pos["PC"]
        # PB = Puntos Totales Obtenidos - (Partidos Ganados * 4 + Empatados * 2)
        pos["PB"] = pos["PTS"] - (pos["PG"] * 4 + pos["PE"] * 2)
        if pos["PB"] < 0:
            pos["PB"] = 0

    tabla = pd.DataFrame(list(posiciones.values()))
    
    # Criterio Desempate: Se agrega el PB antes de DIF
    tabla = tabla.sort_values(by=["PTS", "PB", "DIF", "PF"], ascending=[False, False, False, False]).reset_index(drop=True)
    
    # Agregar columna de Posición
    tabla.insert(0, "Pos.", range(1, len(tabla) + 1))
    
    # Reordenar las columnas visiblemente
    columnas_orden = ["Pos.", "Equipo", "PTS", "PJ", "PG", "PE", "PP", "PB", "PF", "PC", "DIF"]
    tabla = tabla[columnas_orden]
    
    return tabla


def predecir_resultado(local, visitante, tabla_posiciones, df_jugados, parse_resultado):
    """
    Predice el resultado de un partido entre local y visitante.
    Considera rendimiento promedio ponderado por recencia, dificultad de rivales,
    historial directo entre equipos y ventaja de localía.

    Args:
        local (str): nombre del equipo local.
        visitante (str): nombre del equipo visitante.
        tabla_posiciones (DataFrame): incluye columnas ['Equipo', 'PJ', 'PTS'].
        df_jugados (DataFrame): incluye columnas ['Local', 'Visitante', 'ResultadoL', 'ResultadoV'].
        parse_resultado (function): Función para parsear los resultados.

    Returns:
        dict or None: Diccionario con predicción, confianza y factores, o None si no se puede predecir.
    """

    # --- Constantes del modelo ---
    RECENCY_DECAY = 0.85       # Cada partido anterior pierde ~15% de peso
    HOME_ADVANTAGE = 1.05      # 5% bonus al equipo local
    SOS_SENSITIVITY = 0.15     # Sensibilidad del ajuste por dificultad de rivales

    # --- Paso 1: Filtrar partidos jugados de cada equipo ---
    partidos_local = df_jugados[(df_jugados["Local"] == local) | (df_jugados["Visitante"] == local)].copy()
    partidos_visitante = df_jugados[(df_jugados["Local"] == visitante) | (df_jugados["Visitante"] == visitante)].copy()

    if partidos_local.empty or partidos_visitante.empty:
        return None

    # --- Paso 2: Calcular promedios PF y PC con ponderación por recencia ---
    def calc_promedios(partidos, equipo):
        puntos_favor = []
        puntos_contra = []
        rivales_enfrentados = []

        for _, row in partidos.iterrows():
            es_local_actual = row["Local"] == equipo
            rival = row["Visitante"] if es_local_actual else row["Local"]
            resultado_equipo_str = row["ResultadoL"] if es_local_actual else row["ResultadoV"]
            resultado_rival_str = row["ResultadoV"] if es_local_actual else row["ResultadoL"]

            pf, _ = parse_resultado(resultado_equipo_str)
            pc, _ = parse_resultado(resultado_rival_str)

            if pf is not None and pc is not None:
                puntos_favor.append(pf)
                puntos_contra.append(pc)
                rivales_enfrentados.append(rival)

        # Ponderación por recencia: decaimiento exponencial
        # Los partidos más recientes (últimos en la lista) pesan más
        n = len(puntos_favor)
        if n > 0:
            weights = np.array([RECENCY_DECAY ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()
            pf_avg = float(np.average(puntos_favor, weights=weights))
            pc_avg = float(np.average(puntos_contra, weights=weights))
            pf_std = float(np.sqrt(np.average((np.array(puntos_favor) - pf_avg) ** 2, weights=weights)))
        else:
            pf_avg, pc_avg, pf_std = 0, 0, 0

        return pf_avg, pc_avg, rivales_enfrentados, pf_std

    pf_local, pc_local, rivales_local_enfrentados, std_local = calc_promedios(partidos_local, local)
    pf_visitante, pc_visitante, rivales_visitante_enfrentados, std_visitante = calc_promedios(partidos_visitante, visitante)

    # --- Paso 3: Ponderar por Dificultad de Rivales (Strength of Schedule - SoS) ---
    equipos_con_partidos = tabla_posiciones[tabla_posiciones['PJ'] > 0]
    if not equipos_con_partidos.empty:
        league_avg_tpg = float(np.mean(equipos_con_partidos['PTS'] / equipos_con_partidos['PJ']))
    else:
        league_avg_tpg = 2.2

    def get_avg_tpg_of_opponents(rivales_list, tabla_pos, default_tpg):
        tpg_values = []
        for r_name in rivales_list:
            fila_rival = tabla_pos[tabla_pos["Equipo"] == r_name]
            if not fila_rival.empty and fila_rival["PJ"].values[0] > 0:
                tpg_values.append(fila_rival["PTS"].values[0] / fila_rival["PJ"].values[0])
        return float(np.mean(tpg_values)) if tpg_values else default_tpg

    tpg_opp_local = get_avg_tpg_of_opponents(rivales_local_enfrentados, tabla_posiciones, league_avg_tpg)
    tpg_opp_visitante = get_avg_tpg_of_opponents(rivales_visitante_enfrentados, tabla_posiciones, league_avg_tpg)

    ajuste_local_raw = 1.0 + (tpg_opp_local - league_avg_tpg) * SOS_SENSITIVITY
    ajuste_visitante_raw = 1.0 + (tpg_opp_visitante - league_avg_tpg) * SOS_SENSITIVITY
    ajuste_local = max(0.7, min(ajuste_local_raw, 1.3))
    ajuste_visitante = max(0.7, min(ajuste_visitante_raw, 1.3))

    # --- Paso 4: Historial directo como sesgo ---
    historial = df_jugados[
        ((df_jugados["Local"] == local) & (df_jugados["Visitante"] == visitante)) |
        ((df_jugados["Local"] == visitante) & (df_jugados["Visitante"] == local))
    ].copy()

    sesgo_local = 0
    sesgo_visitante = 0
    if not historial.empty:
        pf_hist_local_list = []
        pf_hist_visitante_list = []
        for _, row in historial.iterrows():
            pf_l_hist, _ = parse_resultado(row["ResultadoL"])
            pf_v_hist, _ = parse_resultado(row["ResultadoV"])
            if pf_l_hist is not None and pf_v_hist is not None:
                if row["Local"] == local:
                    pf_hist_local_list.append(pf_l_hist)
                    pf_hist_visitante_list.append(pf_v_hist)
                else:
                    pf_hist_local_list.append(pf_v_hist)
                    pf_hist_visitante_list.append(pf_l_hist)
        
        if pf_hist_local_list:
            sesgo_local = np.mean(pf_hist_local_list) - pf_local
        if pf_hist_visitante_list:
            sesgo_visitante = np.mean(pf_hist_visitante_list) - pf_visitante

    # --- Paso 5: Predicción Final ---
    adj_pf_local = pf_local * ajuste_local
    adj_pc_local = pc_local / ajuste_local
    adj_pf_visitante = pf_visitante * ajuste_visitante
    adj_pc_visitante = pc_visitante / ajuste_visitante

    pred_local_base = (adj_pf_local + adj_pc_visitante) / 2
    pred_visitante_base = (adj_pf_visitante + adj_pc_local) / 2

    # --- Paso 5b: Factor de Localía ---
    pred_local_base *= HOME_ADVANTAGE
    pred_visitante_base /= HOME_ADVANTAGE

    # Aplicar sesgo histórico
    pred_l_final = round(pred_local_base + sesgo_local)
    pred_v_final = round(pred_visitante_base + sesgo_visitante)
    pred_l_final = max(0, int(pred_l_final))
    pred_v_final = max(0, int(pred_v_final))

    # --- Paso 6: Cálculo de Confianza ---
    n_local = len(partidos_local)
    n_visitante = len(partidos_visitante)
    min_partidos = min(n_local, n_visitante)

    # Base por cantidad de datos (0-50 pts)
    conf_datos = min(min_partidos * 10, 50)

    # Bonus por historial directo (0-20 pts)
    conf_hist = min(len(historial) * 10, 20)

    # Bonus por consistencia: baja varianza = alta confianza (0-30 pts)
    avg_std = (std_local + std_visitante) / 2
    # Normalizar: std=0 → 30pts, std=20 → 0pts
    conf_consist = max(0, int(30 * (1 - min(avg_std / 20, 1))))

    confianza_pct = min(conf_datos + conf_hist + conf_consist, 100)
    if confianza_pct >= 70:
        nivel_confianza = "Alta"
    elif confianza_pct >= 40:
        nivel_confianza = "Media"
    else:
        nivel_confianza = "Baja"

    return {
        "pred_local": pred_l_final,
        "pred_visitante": pred_v_final,
        "confianza": confianza_pct,
        "nivel_confianza": nivel_confianza,
        "factores": {
            "sos_local": round(ajuste_local, 3),
            "sos_visitante": round(ajuste_visitante, 3),
            "localía": HOME_ADVANTAGE,
            "sesgo_hist_local": round(sesgo_local, 1),
            "sesgo_hist_visitante": round(sesgo_visitante, 1),
            "partidos_local": n_local,
            "partidos_visitante": n_visitante,
            "historial_directo": len(historial),
        }
    }


st.set_page_config(page_title="Rugby Juveniles", layout="wide")

# Estilo para centrar el contenido y limitar el ancho al 80%
# Unified CSS for premium mobile-first experience
st.markdown("""
    <style>
        /* Streamlit padding — enough to clear the top bar */
        .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 0 !important;
        }
        header[data-testid="stHeader"] {
            height: 2.5rem !important;
        }
        /* Reduce gap between elements globally */
        .stElementContainer {
            margin-bottom: -0.25rem;
        }

        /* 5. Container responsive */
        .main-container {
            max-width: 95%;
            margin: 0 auto;
            padding: 0.25rem 0.5rem;
        }
        @media (min-width: 768px) {
            .main-container {
                max-width: 80%;
                padding: 1rem 1rem;
            }
        }
        
        /* Category selector — radio as compact pills */
        div[role="radiogroup"] {
            display: flex !important;
            justify-content: center !important;
            flex-wrap: wrap !important;
            gap: 0.35rem !important;
            margin-bottom: 0.5rem !important;
        }

        div[role="radiogroup"] label {
            min-height: 32px;
            min-width: auto;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.2rem 0.7rem;
            font-size: 0.82rem;
            cursor: pointer;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.15) !important;
            background: rgba(255,255,255,0.05) !important;
            transition: all 0.15s ease;
        }
        div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.12) !important;
        }
        /* Hide the radio circle */
        div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        /* Selected pill */
        div[role="radiogroup"] label[data-checked="true"],
        div[role="radiogroup"] label:has(input:checked) {
            background: rgba(255, 75, 75, 0.2) !important;
            border-color: rgba(255, 75, 75, 0.6) !important;
            color: #ff4b4b !important;
            font-weight: 700;
        }



        /* 10. Streamlit Tabs — más grandes y táctiles */
        button[data-baseweb="tab"] {
            min-height: 44px;
            padding: 0.5rem 1rem;
            font-size: 0.95rem;
            font-weight: 500;
        }
        @media (max-width: 480px) {
            button[data-baseweb="tab"] {
                padding: 0.4rem 0.6rem;
                font-size: 0.8rem;
            }
        }

        /* 10. Multiselect chips — touch friendly */
        span[data-baseweb="tag"] {
            min-height: 32px;
            padding: 0.25rem 0.5rem;
        }

        /* 10. Expanders — touch target */
        details summary {
            min-height: 44px;
            display: flex;
            align-items: center;
            cursor: pointer;
        }

        /* 10. Reduced motion */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation: none !important;
                transition: none !important;
            }
        }

        /* 10. Base typography */
        .stMarkdown, .stText, p, span {
            font-size: 1rem;
            line-height: 1.5;
        }

        /* 7. Match Cards (La bola de cristal) */
        .match-card {
            background: linear-gradient(135deg, rgba(30,30,50,0.6) 0%, rgba(40,40,70,0.4) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }
        .match-date {
            font-size: 0.8rem;
            color: rgba(255,255,255,0.5);
            margin-bottom: 0.5rem;
        }
        .match-teams {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
        }
        .team-name {
            font-size: 1.05rem;
            font-weight: 600;
            flex: 1;
        }
        .team-local { text-align: left; }
        .team-visitante { text-align: right; }
        .match-score {
            font-size: 1.6rem;
            font-weight: 800;
            text-align: center;
            min-width: 100px;
            letter-spacing: 2px;
        }
        .score-favorito { color: #2ecc71; }
        .score-perdedor { color: rgba(255,255,255,0.5); }
        .score-empate { color: #f39c12; }
        .match-bar-container {
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.08);
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }
        .match-bar-local {
            height: 100%;
            border-radius: 3px 0 0 3px;
            float: left;
        }
        .match-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            color: rgba(255,255,255,0.45);
        }
        .conf-badge {
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .conf-alta { background: rgba(46,204,113,0.15); color: #2ecc71; }
        .conf-media { background: rgba(241,196,15,0.15); color: #f1c40f; }
        .conf-baja { background: rgba(231,76,60,0.15); color: #e74c3c; }

        @media (max-width: 480px) {
            .match-card {
                padding: 0.8rem 1rem;
            }
            .team-name {
                font-size: 0.85rem;
            }
            .match-score {
                font-size: 1.2rem;
                min-width: 70px;
            }
            .match-date {
                font-size: 0.75rem;
            }
            .match-footer {
                font-size: 0.7rem;
                flex-wrap: wrap;
                gap: 0.3rem;
            }
        }

        /* 8. Result Cards (Análisis por equipo - Resultados Jugados) */
        .result-card {
            padding: 0.6rem 1rem;
            margin-bottom: 0.4rem;
            border-radius: 8px;
            border-left: 4px solid;
            background: rgba(255,255,255,0.03);
        }
        .resultado-ganado { border-left-color: #2ecc71; }
        .resultado-perdido { border-left-color: #e74c3c; }
        .resultado-empate { border-left-color: #95a5a6; }
        .result-meta {
            font-size: 0.75rem;
            color: rgba(255,255,255,0.5);
            margin-bottom: 0.2rem;
        }
        .result-teams {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.95rem;
        }
        .result-score {
            font-weight: 700;
            font-size: 1.1rem;
            min-width: 60px;
            text-align: center;
        }
        .result-equipo { font-weight: 600; flex: 1; }
        .result-rival { flex: 1; text-align: right; color: rgba(255,255,255,0.7); }

        @media (max-width: 480px) {
            .result-teams { font-size: 0.85rem; }
            .result-score { font-size: 1rem; min-width: 50px; }
        }

        /* Streamlit Tabs — prominent navigation bar */
        div[data-testid="stTabs"] {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 4px;
            margin: 0.5rem 0 1.5rem 0;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"] {
            border-radius: 8px !important;
            margin: 0 !important;
            border: none !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            color: rgba(255,255,255,0.5) !important;
            flex: 1; 
            min-width: 90px;
            height: 40px;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #ffffff !important;
            background-color: rgba(255,255,255,0.05) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: rgba(255, 75, 75, 0.15) !important;
            color: #ff4b4b !important;
            font-weight: 700 !important;
        }
        /* Hide the default underline */
        div[data-baseweb="tab-highlight"] {
            display: none !important;
        }

        .na-card {
            background: rgba(30,30,50,0.3);
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            color: rgba(255,255,255,0.4);
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.6rem;
            margin: 1rem 0;
        }
        @media (min-width: 768px) {
            .metrics-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.7rem 0.4rem;
            text-align: center;
        }
        .metric-card-label {
            font-size: 0.7rem;
            color: rgba(255,255,255,0.5);
            margin-bottom: 0.2rem;
            text-transform: uppercase;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-card-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)


# --- Conexión a Google Sheets ---
# (get_gspread_client se cachea y solo se ejecuta una vez o cuando sea necesario)
gs_client = get_gspread_client()

# --- Selección de División (Año de Nacimiento o Torneos) ---
# Obtener dinámicamente las pestañas disponibles
available_years_str = get_available_birth_years(gs_client)

# Filtrar las hojas de tarjetas (las que terminan en 'T' si su contraparte existe)
options_for_selectbox = [y for y in available_years_str if not (y.endswith("T") and y[:-1] in available_years_str)]

# Función para ordenar: los numéricos primero en orden descendente, y luego el resto alfabéticamente
def custom_sort(s):
    import re
    # Extraer el primer número que aparezca en el string para ordenar (ej. "2010 Clausura" -> 2010, "M19" -> 19)
    match = re.search(r"(\d+)", s)
    if match:
        return (-int(match.group(1)), s)
    return (0, s)

options_for_selectbox = sorted(options_for_selectbox, key=custom_sort)

# Año por defecto (ej. el más reciente o uno común)
default_year = "2010" if "2010" in options_for_selectbox else (options_for_selectbox[0] if options_for_selectbox else None)

if not default_year:
    st.error("No se pudieron cargar las divisiones desde Google Sheets y no hay fallback. Verifica la configuración.")
    st.stop()

# --- Selección de División — Pill Buttons ---
options_str = [str(opt) for opt in options_for_selectbox]
default_year_str = str(default_year)

# Inicializar session_state
if "ano_nac_seleccionado_str" not in st.session_state:
    st.session_state["ano_nac_seleccionado_str"] = default_year_str

ano_nac_seleccionado_str = st.session_state["ano_nac_seleccionado_str"]

# Selector funcional (radio estilizado como pills por CSS)
selected = st.radio(
    "cat", options=options_str,
    index=options_str.index(ano_nac_seleccionado_str) if ano_nac_seleccionado_str in options_str else 0,
    horizontal=True, key="pill_selector",
    label_visibility="collapsed"
)
if selected != ano_nac_seleccionado_str:
    st.session_state["ano_nac_seleccionado_str"] = selected
    st.rerun()

# Reinicializar clubes_checklist si cambió la división
if "estado_anno_anterior" not in st.session_state or st.session_state["estado_anno_anterior"] != ano_nac_seleccionado_str:
    st.session_state["clubes_checklist"] = {}
    st.session_state["estado_anno_anterior"] = ano_nac_seleccionado_str

# --- Carga de Datos para la División Seleccionada ---
if ano_nac_seleccionado_str:
    df_raw_data = get_division_data(gs_client, ano_nac_seleccionado_str)

    if df_raw_data.empty and ano_nac_seleccionado_str in available_years_str:
        st.warning(f"No hay datos disponibles en la planilla para el año {ano_nac_seleccionado_str} o hubo un error al cargar.")
    elif not df_raw_data.empty:
        expected_cols = ['Nro.', 'Local', 'ResultadoL', 'ResultadoV', 'Visitante', 'Fecha y Hora', 'Estado']
        if not all(col in df_raw_data.columns for col in expected_cols):
            st.error(f"La planilla para el año {ano_nac_seleccionado_str} no tiene todas las columnas esperadas: {expected_cols}")
        else:
            # Tabs inmediatos sin texto extra
          
            # Creamos las pestañas
            tab1, tab_res, tab3, tab2, tab5, tab4 = st.tabs([
                "Posiciones", 
                "Resultados",
                "Equipos", 
                "Pendientes", 
                "Disciplina",
                "Predicciones"
            ])
            # --- TABLA DE POSICIONES ---
            with tab1:
                df_jugados = df_raw_data[df_raw_data["Estado"].str.startswith("Cerrado")].copy()
                # Tabla directa sin subheader redundante
                tabla_posiciones = procesar_partidos(df_jugados)

                # Estilizar la tabla (Top 6: verde para Top 4, amarillo para 5-6)
                def color_clasificacion(row):
                    if row.name < 4:  # Índices 0-3: Top 4 clasificación directa
                        return ['background-color: rgba(46, 204, 113, 0.15)'] * len(row)
                    elif row.name < 6:  # Índices 4-5: zona de repechaje
                        return ['background-color: rgba(241, 196, 15, 0.10)'] * len(row)
                    return [''] * len(row)

                tabla_estilizada = tabla_posiciones.style.apply(color_clasificacion, axis=1)

                # Calcular altura dinámica
                filas = len(tabla_posiciones)
                altura = int(filas * 35 + 40)

                st.dataframe(tabla_estilizada, hide_index=True, use_container_width=True, height=altura)

            # --- RESULTADOS POR FECHA ---
            with tab_res:
                st.subheader("Resultados por Fecha")
                
                # Asegurar conversión a datetime
                if "Fecha_dt" not in df_raw_data.columns:
                    df_raw_data["Fecha_dt"] = pd.to_datetime(df_raw_data["Fecha y Hora"], errors="coerce", dayfirst=True)
                
                # Crear etiquetas para agrupar por día (ej: "Sáb 26/04")
                dias_abrev_res = {
                    "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mié",
                    "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sáb", "Sunday": "Dom"
                }
                
                def fmt_fecha_grupo(dt):
                    if pd.isna(dt): return "Sin Fecha"
                    dia = dias_abrev_res.get(dt.strftime('%A'), "")
                    return f"{dia} {dt.strftime('%d/%m')}"

                df_raw_data["Fecha_Grupo"] = df_raw_data["Fecha_dt"].apply(fmt_fecha_grupo)
                
                # Obtener opciones únicas ordenadas cronológicamente por la fecha real
                opciones_fecha = df_raw_data.sort_values("Fecha_dt")["Fecha_Grupo"].unique().tolist()
                
                if opciones_fecha:
                    # Seleccionar por defecto la última fecha con resultados cerrados
                    df_cerrados = df_raw_data[df_raw_data["Estado"].str.startswith("Cerrado", na=False)]
                    default_idx = len(opciones_fecha) - 1
                    if not df_cerrados.empty:
                        ultima_fecha_valida = fmt_fecha_grupo(df_cerrados.sort_values("Fecha_dt")["Fecha_dt"].iloc[-1])
                        if ultima_fecha_valida in opciones_fecha:
                            default_idx = opciones_fecha.index(ultima_fecha_valida)

                    st.caption("Seleccioná la fecha:")
                    fecha_sel = st.radio(
                        "Seleccioná la fecha de los partidos:", 
                        opciones_fecha, 
                        index=default_idx,
                        horizontal=True,
                        key="fecha_selector_radio",
                        label_visibility="collapsed"
                    )
                    
                    df_ronda = df_raw_data[df_raw_data["Fecha_Grupo"] == fecha_sel].copy()
                    
                    st.markdown(f"**Partidos del {fecha_sel}**")
                    
                    for _, row in df_ronda.iterrows():
                        loc = row["Local"]
                        vis = row["Visitante"]
                        est = row["Estado"]
                        resL = row["ResultadoL"]
                        resV = row["ResultadoV"]
                        fecha_h = row["Fecha y Hora"]
                        
                        if est.startswith("Cerrado"):
                            # Partido Jugado
                            pfL, ptL = parse_resultado(resL)
                            pfV, ptV = parse_resultado(resV)
                            
                            ptL_val = int(ptL) if ptL is not None else 0
                            ptV_val = int(ptV) if ptV is not None else 0
                            
                            # Identificar ganador
                            ganador_l = pfL > pfV if pfL is not None and pfV is not None else False
                            ganador_v = pfV > pfL if pfL is not None and pfV is not None else False
                            
                            res_class = "resultado-empate"
                            score_disp = f"{int(pfL) if pfL is not None else 0} - {int(pfV) if pfV is not None else 0}"
                            meta_disp = "Finalizado"
                            
                            # Formatear nombres con puntos y negrita si ganó
                            loc_html = f"<b>{loc}</b> <span style='opacity: 0.6; font-size: 0.8rem;'>({ptL_val})</span>" if ganador_l else f"{loc} <span style='opacity: 0.6; font-size: 0.8rem;'>({ptL_val})</span>"
                            vis_html = f"<b>{vis}</b> <span style='opacity: 0.6; font-size: 0.8rem;'>({ptV_val})</span>" if ganador_v else f"{vis} <span style='opacity: 0.6; font-size: 0.8rem;'>({ptV_val})</span>"
                        else:
                            # Partido Pendiente
                            res_class = "na-card"
                            score_disp = "vs"
                            meta_disp = fecha_h
                            loc_html = loc
                            vis_html = vis
                        
                        st.markdown(f"""
                        <div class="result-card {res_class if est.startswith('Cerrado') else ''}" style="{'background: rgba(255,255,255,0.02); border-left: 4px solid rgba(255,255,255,0.1);' if not est.startswith('Cerrado') else ''}">
                            <div class="result-meta">{meta_disp}</div>
                            <div class="result-teams">
                                <span class="result-equipo" style="text-align: left; flex: 2;">{loc_html}</span>
                                <span class="result-score" style="flex: 1;">{score_disp}</span>
                                <span class="result-rival" style="text-align: right; flex: 2;">{vis_html}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No se encontró una columna de 'Instancia' o 'Nro.' para agrupar los partidos por fecha.")

            # --- PARTIDOS PENDIENTES ---
            with tab2:
                st.subheader("Partidos pendientes")

                df_pendientes = df_raw_data[df_raw_data["Estado"] == "Pendiente"].copy()
                df_pendientes["Fecha_dt"] = pd.to_datetime(df_pendientes["Fecha y Hora"], errors="coerce", dayfirst=True)
                # Días abreviados para compactar tabla en mobile
                dias_abrev = {
                    "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mié",
                    "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sáb", "Sunday": "Dom"
                }
                df_pendientes["Fecha"] = df_pendientes["Fecha_dt"].apply(
                    lambda x: f"{dias_abrev.get(x.strftime('%A'), '')} {x.strftime('%d/%m %H:%M')}" if pd.notna(x) else "-"
                )
                clubes_locales = df_pendientes["Local"].unique()
                clubes_visitantes = df_pendientes["Visitante"].unique()
                clubes_unicos = sorted(set(clubes_locales) | set(clubes_visitantes))

                # 1. Filtro Inteligente Nativo
                clubes_seleccionados = st.multiselect(
                    "Filtrar por clubes (dejá vacío para ver todos):", 
                    options=clubes_unicos, 
                    default=[]
                )

                if not clubes_seleccionados:
                    clubes_activos = clubes_unicos
                else:
                    clubes_activos = clubes_seleccionados

                df_filtrado = df_pendientes[
                    df_pendientes["Local"].isin(clubes_activos) |
                    df_pendientes["Visitante"].isin(clubes_activos)
                ].sort_values("Fecha_dt").copy()

                # 2. Panel de Métricas Rápidas (KPIs)
                col_metric_1, col_metric_2 = st.columns(2)
                with col_metric_1:
                    st.metric("Partidos Pendientes Totales", len(df_filtrado))
                with col_metric_2:
                    if not df_filtrado.empty and not df_filtrado["Fecha_dt"].dropna().empty:
                        prox_fecha = df_filtrado["Fecha_dt"].dropna().min()
                        prox_partido_str = prox_fecha.strftime('%d/%m/%Y %H:%M')
                    else:
                        prox_partido_str = "N/A"
                    st.metric("Próximo Partido", prox_partido_str)

                # 3. Mejoras Visuales en el DataFrame
                if not df_filtrado.empty:
                    df_mostrar = df_filtrado.rename(columns={
                        "Local": "🏠 Local", 
                        "Visitante": "✈️ Visitante"
                    })
                    columnas_mostrar = ["Fecha", "🏠 Local", "✈️ Visitante"]
                else:
                    df_mostrar = pd.DataFrame(columns=["Fecha", "🏠 Local", "✈️ Visitante"])
                    columnas_mostrar = df_mostrar.columns.tolist()

                # Calcular altura dinámica
                filas = len(df_mostrar)
                altura = min(int(filas * 35 + 40), 800) if filas > 0 else 100

                st.dataframe(
                    df_mostrar[columnas_mostrar],
                    hide_index=True,
                    use_container_width=True,
                    height=altura
                )

            # ---------- Análisis por equipo ----------
            with tab3:
                # Subheader eliminado para ahorrar espacio

                equipos = sorted(tabla_posiciones["Equipo"].unique())
                # Seleccionar "UNIVERSITARIO" por defecto si existe, si no el primero.
                default_index = equipos.index("UNIVERSITARIO") if "UNIVERSITARIO" in equipos else 0
                st.caption("Seleccioná un equipo:")
                equipo_sel = st.radio(
                    "Seleccioná un equipo", 
                    equipos, 
                    index=default_index, 
                    horizontal=True, 
                    key="equipo_selector_radio",
                    label_visibility="collapsed"
                )

                if equipo_sel:
                    # Partidos jugados por el equipo
                    jugados_equipo = df_jugados[(df_jugados["Local"] == equipo_sel) | (df_jugados["Visitante"] == equipo_sel)].copy()
                    
                    if not jugados_equipo.empty: # Verificar si el equipo tiene partidos
                        # Parseo de resultados
                        jugados_equipo["Puntos Equipo"] = jugados_equipo.apply(
                            lambda row: parse_resultado(row["ResultadoL"] if row["Local"] == equipo_sel else row["ResultadoV"])[0],
                            axis=1
                        ).fillna(0)
                        jugados_equipo["Puntos Rival"] = jugados_equipo.apply(
                            lambda row: parse_resultado(row["ResultadoV"] if row["Local"] == equipo_sel else row["ResultadoL"])[0],
                            axis=1
                        ).fillna(0)
                        jugados_equipo["Pts Torneo Equipo"] = jugados_equipo.apply(
                            lambda row: parse_resultado(row["ResultadoL"] if row["Local"] == equipo_sel else row["ResultadoV"])[1],
                            axis=1
                        ).fillna(0)
                        jugados_equipo["Pts Torneo Rival"] = jugados_equipo.apply(
                            lambda row: parse_resultado(row["ResultadoV"] if row["Local"] == equipo_sel else row["ResultadoL"])[1],
                            axis=1
                        ).fillna(0)

                        # Cálculo de victorias, empates, derrotas
                        victorias = (jugados_equipo["Puntos Equipo"] > jugados_equipo["Puntos Rival"]).sum()
                        empates = (jugados_equipo["Puntos Equipo"] == jugados_equipo["Puntos Rival"]).sum()
                        derrotas = (jugados_equipo["Puntos Equipo"] < jugados_equipo["Puntos Rival"]).sum()
                        total = len(jugados_equipo)
                        
                        puntos_totales_torneo_equipo = jugados_equipo["Pts Torneo Equipo"].sum()
                        
                        prom_puntos_favor = jugados_equipo['Puntos Equipo'].mean() if total > 0 else 0
                        prom_puntos_contra = jugados_equipo['Puntos Rival'].mean() if total > 0 else 0
                        dif_prom_partido = prom_puntos_favor - prom_puntos_contra

                        # Flujo vertical — métricas primero, gráfico después
                        st.markdown(f"#### Estadísticas de {equipo_sel}")
                        
                        # Grilla de métricas responsiva (2 col mobile, 4 col desktop)
                        pb_calc = int(puntos_totales_torneo_equipo - (victorias * 4 + empates * 2)) if total > 0 else 0
                        ganados_str = f"{victorias} ({victorias/total:.0%})" if total > 0 else "0"
                        perdidos_str = f"{derrotas} ({derrotas/total:.0%})" if total > 0 else "0"
                        empatados_str = f"{empates} ({empates/total:.0%})" if total > 0 else "0"
                        prom_str = f"{prom_puntos_favor:.1f} / {prom_puntos_contra:.1f}"
                        dif_str = f"{dif_prom_partido:+.1f}"

                        st.markdown(f"""
                        <div class="metrics-grid">
                            <div class="metric-card"><div class="metric-card-label">Jugados</div><div class="metric-card-value">{total}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Pts. Torneo</div><div class="metric-card-value">{int(puntos_totales_torneo_equipo)}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Ganados</div><div class="metric-card-value">{ganados_str}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Perdidos</div><div class="metric-card-value">{perdidos_str}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Empatados</div><div class="metric-card-value">{empatados_str}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Pts. Bonus</div><div class="metric-card-value">{pb_calc}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Prom. PF/PC</div><div class="metric-card-value">{prom_str}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Dif. Prom.</div><div class="metric-card-value">{dif_str}</div></div>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander("Distribución de Resultados", expanded=False):
                            if total > 0: 
                                # Gráfico interactivo con px.pie en lugar de plt
                                df_pie = pd.DataFrame({
                                    "Resultado": ["Ganados", "Empatados", "Perdidos"],
                                    "Cantidad": [victorias, empates, derrotas]
                                })
                                # Filtramos para que no salgan porciones con valor 0
                                df_pie = df_pie[df_pie["Cantidad"] > 0]
                                
                                fig1 = px.pie(
                                    df_pie, 
                                    values="Cantidad", 
                                    names="Resultado",
                                    color="Resultado",
                                    color_discrete_map={"Ganados": "#2ecc71", "Empatados": "#95a5a6", "Perdidos": "#e74c3c"}
                                )
                                fig1.update_traces(
                                    textinfo='percent+value',
                                    hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
                                )
                                fig1.update_layout(
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                                    margin=dict(t=20, b=20, l=20, r=20),
                                    dragmode=False
                                )
                                st.plotly_chart(fig1, use_container_width=True, config={
                                    'displayModeBar': False,
                                    'scrollZoom': False,
                                    'staticPlot': False,
                                    'responsive': True
                                })
                            else:
                                st.info("No hay partidos jugados para mostrar en el gráfico.")
                    else:
                        st.info(f"No se encontraron partidos jugados para {equipo_sel}.")

                    # ---------- Evolución de puntos ----------
                    jugados_equipo["Fecha"] = pd.to_datetime(jugados_equipo["Fecha y Hora"], errors="coerce", dayfirst=True)
                    jugados_equipo = jugados_equipo.sort_values("Fecha")

                    jugados_equipo["Rival"] = jugados_equipo.apply(
                        lambda row: row["Visitante"] if row["Local"] == equipo_sel else row["Local"],
                        axis=1
                    )
                    jugados_equipo["Etiqueta"] = jugados_equipo.apply(
                        lambda row: f"{row['Rival']} ({row['Fecha'].strftime('%d/%m')})", axis=1
                    )
                    # Columna customizada para alimentar el tooltip con el resultado real
                    jugados_equipo["Resultado"] = jugados_equipo.apply(
                        lambda row: f"{row['ResultadoL']} - {row['ResultadoV']}", axis=1
                    )

                    # Crear DataFrame para gráfico de líneas interactivo
                    df_line = pd.melt(
                        jugados_equipo[["Etiqueta", "Puntos Equipo", "Puntos Rival", "Rival", "Resultado", "Fecha"]],
                        id_vars=["Etiqueta", "Rival", "Resultado", "Fecha"],
                        var_name="Tipo",
                        value_name="Puntos"
                    )

                    # Renombrar los tipos para la leyenda del gráfico
                    df_line["Tipo"] = df_line["Tipo"].map({
                        "Puntos Equipo": "A favor",
                        "Puntos Rival": "En contra"
                    })

                    with st.expander("Evolución de Puntos por Partido", expanded=False):
                        # Gráfico de líneas con Plotly
                        fig2 = px.line(
                            df_line,
                            x="Etiqueta",
                            y="Puntos",
                            color="Tipo",
                            markers=True,
                            text="Puntos",
                            color_discrete_map={"A favor": "#3498db", "En contra": "#e74c3c"},
                            hover_data={"Etiqueta": False, "Tipo": False, "Rival": True, "Resultado": True}
                        )
                        
                        # Tooltip personalizado configurando el customdata entregado por hover_data
                        fig2.update_traces(
                            textposition="top center",
                            hovertemplate='<b>%{x}</b><br>Puntos %{legendgroup}: %{y}<br>Rival: %{customdata[0]}<br>Resultado: %{customdata[1]}<extra></extra>'
                        )

                        # Mejoras estéticas y layout de Plotly
                        fig2.update_layout(
                            title=dict(text=f"Evolución de Puntos por Partido - {equipo_sel}"),
                            xaxis=dict(
                                tickangle=-45,
                                tickfont=dict(size=10),
                                title=None
                            ),
                            yaxis_title="Puntos",
                            legend_title="Tipo",
                            hovermode="x unified",
                            margin=dict(t=50, b=40, l=40, r=20),
                            dragmode=False
                        )

                        st.plotly_chart(fig2, use_container_width=True, config={
                            'displayModeBar': False,
                            'scrollZoom': False,
                            'staticPlot': False,
                            'responsive': True
                        })

                    # --- Sección de Resultados Jugados ---
                    st.markdown("#### Resultados Jugados")

                    for _, row in jugados_equipo.iterrows():
                        rival = row["Rival"]
                        pf = row["Puntos Equipo"]
                        pc = row["Puntos Rival"]
                        fecha = row["Fecha"]
                        es_local = row["Local"] == equipo_sel
                        condicion = "Local" if es_local else "Visitante"
                        
                        if pf > pc:
                            resultado_class = "resultado-ganado"
                        elif pf < pc:
                            resultado_class = "resultado-perdido"
                        else:
                            resultado_class = "resultado-empate"
                        
                        dias_abrev_rc = {"Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mié", "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sáb", "Sunday": "Dom"}
                        fecha_str = f"{dias_abrev_rc.get(fecha.strftime('%A'), '')} {fecha.strftime('%d/%m')}" if pd.notna(fecha) else "-"

                        pts_e = row["Pts Torneo Equipo"]
                        pts_r = row["Pts Torneo Rival"]
                        
                        ganador_e = pf > pc
                        ganador_r = pc > pf
                        
                        equipo_html = f"<b>{equipo_sel}</b>" if ganador_e else equipo_sel
                        rival_html = f"<b>{rival}</b>" if ganador_r else rival
                        
                        st.markdown(f"""
                        <div class="result-card {resultado_class}">
                            <div class="result-meta">{fecha_str} · {condicion}</div>
                            <div class="result-teams">
                                <span class="result-equipo" style="text-align: left; flex: 2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    {equipo_html} <span style="opacity: 0.6; font-size: 0.8rem;">({int(pts_e)})</span>
                                </span>
                                <span class="result-score" style="flex: 1;">{int(pf)} - {int(pc)}</span>
                                <span class="result-rival" style="text-align: right; flex: 2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    <span style="opacity: 0.6; font-size: 0.8rem;">({int(pts_r)})</span> {rival_html}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Definir siempre esta variable, esté o no en partidos pendientes
                    equipo_seleccionado = equipo_sel
                    # ---------- Mostrar partidos pendientes ----------
                    pendientes_equipo = df_pendientes[
                        (df_pendientes["Local"] == equipo_sel) | (df_pendientes["Visitante"] == equipo_sel)
                    ].copy()

                    if not pendientes_equipo.empty:
                        # Procesamiento de datos para las tarjetas de pendientes
                        pendientes_equipo["Fecha_dt"] = pd.to_datetime(pendientes_equipo["Fecha y Hora"], errors="coerce", dayfirst=True)
                        dias_abrev_pe = {"Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mié", "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sáb", "Sunday": "Dom"}
                        pendientes_equipo["Fecha"] = pendientes_equipo["Fecha_dt"].apply(
                            lambda x: f"{dias_abrev_pe.get(x.strftime('%A'), '')} {x.strftime('%d/%m %H:%M')}" if pd.notna(x) else "-"
                        )
                        pendientes_equipo["Rival"] = pendientes_equipo.apply(
                            lambda row: row["Visitante"] if row["Local"] == equipo_seleccionado else row["Local"],
                            axis=1
                        )
                        pendientes_equipo["Condición"] = pendientes_equipo["Local"].apply(
                            lambda x: "Local" if x == equipo_seleccionado else "Visitante"
                        )
                        pendientes_equipo = pendientes_equipo.sort_values("Fecha_dt")

                        st.markdown("#### Partidos pendientes")

                        for _, row in pendientes_equipo.iterrows():
                            # Reutilizamos el estilo de na-card para pendientes
                            st.markdown(f"""
                            <div class="result-card na-card" style="margin-bottom: 0.5rem; padding: 0.6rem 1rem; background: rgba(30,30,50,0.4); border-left: 4px solid #f1c40f;">
                                <div class="result-meta">{row['Fecha']} · {row['Condición']}</div>
                                <div class="result-teams">
                                    <span class="result-equipo" style="text-align: left; flex: 2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{equipo_seleccionado}</span>
                                    <span class="result-score" style="flex: 1; font-size: 0.9rem; opacity: 0.7;">vs</span>
                                    <span class="result-rival" style="text-align: right; flex: 2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{row['Rival']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Este equipo no tiene partidos pendientes.")

                    # Cargar tarjetas si hay hoja disponible
                    df_tarjetas = get_tarjetas_data(gs_client, ano_nac_seleccionado_str)

                    # Filtrar tarjetas del equipo seleccionado
                    df_tarjetas_equipo = df_tarjetas[df_tarjetas["Equipo"] == equipo_seleccionado]

                    if not df_tarjetas_equipo.empty:
                        st.markdown("#### Incidencias disciplinarias")
                        
                        # Mostrar resumen rápido
                        total_amarillas = df_tarjetas_equipo["Incidencia"].str.contains("amarilla", case=False).sum()
                        total_rojas = df_tarjetas_equipo["Incidencia"].str.contains("roja", case=False).sum()
                        total_azules = df_tarjetas_equipo["Incidencia"].str.contains("azul", case=False).sum()

                        # Grilla de métricas para tarjetas del equipo
                        st.markdown(f"""
                        <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                            <div class="metric-card"><div class="metric-card-label">Amarillas</div><div class="metric-card-value">{total_amarillas}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Rojas</div><div class="metric-card-value">{total_rojas}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Azules</div><div class="metric-card-value">{total_azules}</div></div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Mostrar tabla detallada
                        st.dataframe(
                            df_tarjetas_equipo[["Fecha", "Incidencia", "Instancia", "Rival", "Momento", "Detalle"]],
                            use_container_width=True,
                            hide_index=True,
                            height=min(len(df_tarjetas_equipo) * 35 + 40, 600),  # Altura dinámica
                        )
                    else:
                        st.info("Este equipo no tiene incidencias registradas.")

            # ---------- Predicción de partidos pendientes ----------
            with tab4:        
                    st.subheader("La bola de cristal... puede fallar!")

                    # --- Explicación del modelo ---
                    with st.expander("¿Cómo funciona el modelo?"):
                        st.markdown("""
                        El modelo de predicción combina **4 factores** para estimar el resultado:

                        1. **Promedio Ponderado por Recencia** — Los últimos partidos pesan más que los primeros del año (decay: 0.85).
                        2. **Dificultad de Rivales (SoS)** — Si un equipo le ganó a rivales fuertes, su poder predictivo sube.
                        3. **Historial Directo** — Si los equipos ya jugaron entre sí, el modelo ajusta según ese comportamiento específico.
                        4. **Ventaja de Localía** — El equipo local recibe un bonus del ~5%.

                        **Confianza**: Se calcula según la cantidad de datos disponibles, consistencia de resultados e historial directo.
                        """)

                    if df_pendientes.empty:
                        st.info("No hay partidos pendientes para mostrar predicciones.")
                    elif df_jugados.empty:
                        st.warning("No hay datos de partidos jugados (cerrados). No se pueden generar predicciones detalladas.")
                        st.dataframe(df_pendientes[["Local", "Visitante", "Fecha y Hora"]], hide_index=True, use_container_width=True)
                    else:
                        clubes_locales_pred = df_pendientes["Local"].unique()
                        clubes_visitantes_pred = df_pendientes["Visitante"].unique()
                        clubes_unicos_pred = sorted(set(clubes_locales_pred) | set(clubes_visitantes_pred))

                        # --- Filtro con multiselect ---
                        clubes_pred_activos = st.multiselect(
                            "Filtrar clubes para predicción (dejá vacío para ver todos):",
                            options=clubes_unicos_pred,
                            default=[],
                            key="multiselect_pred_clubes"
                        )

                        # Si no seleccionó ninguno, mostrar todos
                        if not clubes_pred_activos:
                            clubes_pred_activos = clubes_unicos_pred

                        df_pendientes_filtrados = df_pendientes[
                            df_pendientes["Local"].isin(clubes_pred_activos) |
                            df_pendientes["Visitante"].isin(clubes_pred_activos)
                        ].copy()
                        df_pendientes_filtrados["Fecha_dt"] = pd.to_datetime(df_pendientes_filtrados["Fecha y Hora"], errors="coerce", dayfirst=True)
                        df_pendientes_filtrados = df_pendientes_filtrados.sort_values("Fecha_dt")

                        # --- Generar predicciones ---
                        predicciones_list = []
                        for _, row in df_pendientes_filtrados.iterrows():
                            local_eq = row["Local"]
                            visitante_eq = row["Visitante"]
                            resultado = predecir_resultado(local_eq, visitante_eq, tabla_posiciones, df_jugados, parse_resultado)
                            predicciones_list.append({
                                "fecha_raw": row["Fecha y Hora"],
                                "fecha_dt": row.get("Fecha_dt"),
                                "local": local_eq,
                                "visitante": visitante_eq,
                                "resultado": resultado
                            })

                        if not predicciones_list:
                            st.info("No hay predicciones disponibles para mostrar.")
                        else:
                            # --- Panel KPI ---
                            total_pred = len(predicciones_list)
                            pred_validas = [p for p in predicciones_list if p["resultado"] is not None]
                            fav_local = sum(1 for p in pred_validas if p["resultado"]["pred_local"] > p["resultado"]["pred_visitante"])
                            avg_conf = np.mean([p["resultado"]["confianza"] for p in pred_validas]) if pred_validas else 0
                            pct_local_fav = (fav_local / len(pred_validas) * 100) if pred_validas else 0

                            kpi1, kpi2, kpi3 = st.columns(3)
                            kpi1.metric("Predicciones", total_pred)
                            kpi2.metric("Favoritos Locales", f"{pct_local_fav:.0f}%")
                            kpi3.metric("Confianza Promedio", f"{avg_conf:.0f}%")

                            st.markdown("---")

                            # --- Días en español ---
                            dias_es = {
                                "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
                                "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
                            }

                            # CSS removed and unified at start

                            # --- Renderizar Match Cards ---
                            for pred_data in predicciones_list:
                                resultado = pred_data["resultado"]
                                local_eq = pred_data["local"]
                                visitante_eq = pred_data["visitante"]

                                # Formatear fecha
                                fecha_dt = pred_data["fecha_dt"]
                                if pd.notna(fecha_dt):
                                    dia_nombre = dias_es.get(fecha_dt.strftime("%A"), "")
                                    fecha_str = f"{dia_nombre} {fecha_dt.strftime('%d/%m/%Y %H:%M')}"
                                else:
                                    fecha_str = f"{pred_data['fecha_raw']}"

                                if resultado is None:
                                    # Card para predicciones no disponibles
                                    st.markdown(f"""
                                    <div class="na-card">
                                        <div class="match-date">{fecha_str}</div>
                                        <div class="match-teams">
                                            <span class="team-name team-local">{local_eq}</span>
                                            <span class="match-score" style="color: rgba(255,255,255,0.3);">? — ?</span>
                                            <span class="team-name team-visitante">{visitante_eq}</span>
                                        </div>
                                        <div class="match-footer">
                                            <span>Sin datos suficientes para predecir</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    continue

                                pred_l = resultado["pred_local"]
                                pred_v = resultado["pred_visitante"]
                                confianza = resultado["confianza"]
                                nivel = resultado["nivel_confianza"]
                                factores = resultado["factores"]

                                # Determinar favorito y colores de score
                                total_pts = max(pred_l + pred_v, 1)
                                pct_local = pred_l / total_pts * 100

                                if pred_l > pred_v:
                                    score_l_class = "score-favorito"
                                    score_v_class = "score-perdedor"
                                    fav_text = f"← Favorito"
                                elif pred_v > pred_l:
                                    score_l_class = "score-perdedor"
                                    score_v_class = "score-favorito"
                                    fav_text = f"Favorito →"
                                else:
                                    score_l_class = "score-empate"
                                    score_v_class = "score-empate"
                                    fav_text = "Parejo"

                                # Color de barra
                                bar_local_color = "#2ecc71" if pred_l >= pred_v else "#e74c3c"
                                bar_visit_color = "#e74c3c" if pred_l >= pred_v else "#2ecc71"

                                # Clase de confianza
                                if confianza >= 70:
                                    conf_class = "conf-alta"
                                elif confianza >= 40:
                                    conf_class = "conf-media"
                                else:
                                    conf_class = "conf-baja"

                                # Renderizar card
                                st.markdown(f"""
                                <div class="match-card">
                                    <div class="match-date">{fecha_str}</div>
                                    <div class="match-teams">
                                        <span class="team-name team-local">{local_eq}</span>
                                        <span class="match-score">
                                            <span class="{score_l_class}">{pred_l}</span>
                                            <span style="color: rgba(255,255,255,0.25); margin: 0 4px;">—</span>
                                            <span class="{score_v_class}">{pred_v}</span>
                                        </span>
                                        <span class="team-name team-visitante">{visitante_eq}</span>
                                    </div>
                                    <div class="match-bar-container">
                                        <div class="match-bar-local" style="width: {pct_local:.1f}%; background: linear-gradient(90deg, {bar_local_color}, {bar_visit_color});"></div>
                                    </div>
                                    <div class="match-footer">
                                        <span>{fav_text}</span>
                                        <span class="conf-badge {conf_class}">{nivel} ({confianza}%)</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Expander con detalles del modelo
                                with st.expander(f"📊 Detalles: {local_eq} vs {visitante_eq}"):
                                    det1, det2 = st.columns(2)
                                    with det1:
                                        st.markdown(f"""
                                        **🏠 {local_eq}**
                                        - SoS: `{factores['sos_local']}`
                                        - Sesgo histórico: `{factores['sesgo_hist_local']:+.1f}`
                                        - Partidos jugados: `{factores['partidos_local']}`
                                        """)
                                    with det2:
                                        st.markdown(f"""
                                        **✈️ {visitante_eq}**
                                        - SoS: `{factores['sos_visitante']}`
                                        - Sesgo histórico: `{factores['sesgo_hist_visitante']:+.1f}`
                                        - Partidos jugados: `{factores['partidos_visitante']}`
                                        """)
                                    st.caption(f"Localía: {factores['localía']}x | Enfrentamientos directos: {factores['historial_directo']}")

            # ---------- TABLA DE TARJETAS ----------
            with tab5:
                df_tarjetas = get_tarjetas_data(gs_client, ano_nac_seleccionado_str)

                if df_tarjetas.empty:
                    st.info("No hay registros de tarjetas para esta división.")
                else:
                    st.subheader("Tabla de Disciplina")

                    # --- 1. Procesamiento de Datos ---
                    df_tarjetas["Incidencia"] = df_tarjetas["Incidencia"].str.lower().str.strip()
                    
                    # Conteo total para KPIs de toda la división
                    total_y = df_tarjetas[df_tarjetas["Incidencia"].str.contains("amarilla")].shape[0]
                    total_r = df_tarjetas[df_tarjetas["Incidencia"].str.contains("roja")].shape[0]
                    total_b = df_tarjetas[df_tarjetas["Incidencia"].str.contains("azul")].shape[0]

                    # --- 2. Panel de Métricas ---
                    st.markdown(f"""
                    <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                        <div class="metric-card"><div class="metric-card-label">Amarillas</div><div class="metric-card-value">{total_y}</div></div>
                        <div class="metric-card"><div class="metric-card-label">Rojas</div><div class="metric-card-value">{total_r}</div></div>
                        <div class="metric-card"><div class="metric-card-label">Azules</div><div class="metric-card-value">{total_b}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- 3. Filtro por Club ---
                    clubes_con_tarjetas = sorted(df_tarjetas["Equipo"].unique())
                    clubes_sel = st.multiselect(
                        "Filtrar por club (dejar vacío para ver todos):", 
                        options=clubes_con_tarjetas, 
                        default=[]
                    )
                    
                    df_tarjetas_filt = df_tarjetas.copy()
                    if clubes_sel:
                        df_tarjetas_filt = df_tarjetas[df_tarjetas["Equipo"].isin(clubes_sel)]

                    # --- 4. Resumen por Equipo para visualización ---
                    resumen = df_tarjetas_filt.groupby("Equipo")["Incidencia"].value_counts().unstack(fill_value=0).reset_index()
                    
                    # Asegurar que existan las columnas incluso si están en 0
                    mapeo_columnas = {
                        "amarilla": "Amarillas", 
                        "roja": "Rojas", 
                        "azul": "Azules"
                    }
                    
                    for col_original, col_nueva in mapeo_columnas.items():
                        if col_original not in resumen.columns:
                            resumen[col_original] = 0
                        resumen = resumen.rename(columns={col_original: col_nueva})

                    # Ordenar por gravedad (Rojas valen más puntos en un ranking inverso de disciplina)
                    resumen["_score"] = (resumen["Amarillas"] * 1) + (resumen["Rojas"] * 5) + (resumen["Azules"] * 2)
                    resumen = resumen.sort_values("_score", ascending=False).drop(columns=["_score"])

                    # --- 5. Gráfico de Conducta (Plotly) ---
                    st.markdown("#### Conducta por Club")
                    if not resumen.empty:
                        # Preparar datos para Plotly format largo
                        columnas_cards = ["Amarillas", "Azules", "Rojas"]
                        df_plot = resumen.melt(id_vars="Equipo", value_vars=columnas_cards, var_name="Tipo", value_name="Cantidad")
                        
                        fig_cards = px.bar(
                            df_plot, 
                            x="Cantidad", 
                            y="Equipo", 
                            color="Tipo",
                            orientation='h',
                            color_discrete_map={
                                "Amarillas": "#D4AF37", 
                                "Rojas": "#e74c3c", 
                                "Azules": "#3498db"
                            },
                        )
                        fig_cards.update_layout(
                            xaxis={'tickformat': 'd', 'dtick': 1},
                            barmode='stack', 
                            yaxis={'categoryorder':'total ascending'},
                            height=max(min(len(resumen) * 35 + 100, 600), 300),
                            margin=dict(l=20, r=20, t=20, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            dragmode=False
                        )
                        st.plotly_chart(fig_cards, use_container_width=True, config={
                            'displayModeBar': False,
                            'scrollZoom': False,
                            'staticPlot': False,
                            'responsive': True
                        })

                    # --- 6. Tabla Detallada ---
                    st.markdown("#### Detalle por Equipo")
                    st.dataframe(
                        resumen,
                        use_container_width=True,
                        hide_index=True,
                        height=min(len(resumen) * 35 + 40, 400)
                    )

else:
    st.info("Esperando que cargues un archivo CSV con los resultados.")

st.markdown("---")  # Separador visual

st.markdown(
    """
    <div style="font-size: 0.85rem; color: gray; text-align: center;">
        ⚠️ Esta aplicación fue desarrollada con fines recreativos y de entretenimiento.<br>
        Puede contener errores no intencionales y no representa una fuente oficial de datos.<br>
        Para información oficial, por favor consultá directamente la base de datos de la <a href="https://bd.uar.com.ar" target="_blank">UAR</a>.
    </div>
    """,
    unsafe_allow_html=True
)
# Cerramos el div que abrimos antes
st.markdown("</div>", unsafe_allow_html=True)
