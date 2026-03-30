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
    Considera rendimiento promedio, dificultad de rivales y historial entre equipos.

    Args:
        local (str): nombre del equipo local.
        visitante (str): nombre del equipo visitante.
        tabla_posiciones (DataFrame): incluye columnas ['Equipo', 'PJ', 'PTS']. (PTS = Puntos de Torneo)
        df_jugados (DataFrame): incluye columnas ['Local', 'Visitante', 'ResultadoL', 'ResultadoV'].
        parse_resultado (function): Función para parsear los resultados.

    Returns:
        Tuple[int, int] or Tuple[None, None]: puntos esperados (local, visitante).
    """

    # --- Paso 1: Filtrar partidos jugados de cada equipo ---
    partidos_local = df_jugados[(df_jugados["Local"] == local) | (df_jugados["Visitante"] == local)].copy()
    partidos_visitante = df_jugados[(df_jugados["Local"] == visitante) | (df_jugados["Visitante"] == visitante)].copy()

    if partidos_local.empty or partidos_visitante.empty:
        # Si un equipo no tiene historial, no se puede predecir con este método.
        # Podrías devolver (None, None) o una predicción basada en promedios de liga si los tuvieras.
        return None, None

    # --- Paso 2: Calcular promedios PF y PC por equipo ---
    def calc_promedios(partidos, equipo):
        puntos_favor = []
        puntos_contra = []
        rivales_enfrentados = [] # Cambiado de 'rivales' para evitar confusión de nombres

        for _, row in partidos.iterrows():
            es_local_actual = row["Local"] == equipo # Si el 'equipo' fue local en este partido histórico
            rival = row["Visitante"] if es_local_actual else row["Local"]
            resultado_equipo_str = row["ResultadoL"] if es_local_actual else row["ResultadoV"]
            resultado_rival_str = row["ResultadoV"] if es_local_actual else row["ResultadoL"]

            pf, _ = parse_resultado(resultado_equipo_str) # Usamos el primer valor (puntos del partido)
            pc, _ = parse_resultado(resultado_rival_str)  # Usamos el primer valor

            if pf is not None and pc is not None:
                puntos_favor.append(pf)
                puntos_contra.append(pc)
                rivales_enfrentados.append(rival)

        pf_avg = np.mean(puntos_favor) if puntos_favor else 0
        pc_avg = np.mean(puntos_contra) if puntos_contra else 0

        return pf_avg, pc_avg, rivales_enfrentados

    pf_local, pc_local, rivales_local_enfrentados = calc_promedios(partidos_local, local)
    pf_visitante, pc_visitante, rivales_visitante_enfrentados = calc_promedios(partidos_visitante, visitante)

    # --- Paso 3: Ponderar por Dificultad de Rivales (Strength of Schedule - SoS) ---
    
    # Estimar el TPG (Tournament Points Per Game) promedio de la liga
    # Esto es crucial para tener un baseline correcto.
    equipos_con_partidos = tabla_posiciones[tabla_posiciones['PJ'] > 0]
    if not equipos_con_partidos.empty:
        league_avg_tpg = np.mean(equipos_con_partidos['PTS'] / equipos_con_partidos['PJ'])
    else:
        league_avg_tpg = 2.2  # Fallback: Estimación (ajusta según tu liga. Ej: (4pts victoria * 50% + 1pt bonus) ~ 2.0-2.5)

    def get_avg_tpg_of_opponents(rivales_list, tabla_pos, default_tpg_for_missing_rival):
        tpg_values_of_rivals = []
        for r_name in rivales_list:
            fila_rival = tabla_pos[tabla_pos["Equipo"] == r_name]
            if not fila_rival.empty and fila_rival["PJ"].values[0] > 0:
                tpg_rival = fila_rival["PTS"].values[0] / fila_rival["PJ"].values[0]
                tpg_values_of_rivals.append(tpg_rival)
        
        # Si no hay rivales con estadísticas, o la lista de rivales está vacía,
        # se asume que el equipo jugó contra oponentes de fuerza promedio de la liga.
        return np.mean(tpg_values_of_rivals) if tpg_values_of_rivals else default_tpg_for_missing_rival

    # TPG promedio de los oponentes que 'local' ha enfrentado
    tpg_opp_local = get_avg_tpg_of_opponents(rivales_local_enfrentados, tabla_posiciones, league_avg_tpg)
    # TPG promedio de los oponentes que 'visitante' ha enfrentado
    tpg_opp_visitante = get_avg_tpg_of_opponents(rivales_visitante_enfrentados, tabla_posiciones, league_avg_tpg)

    # Sensibilidad del ajuste por SoS. Un valor más bajo significa un ajuste menor.
    # Originalmente tenías 0.5, que es muy alto. Prueba con 0.1 o 0.15.
    sos_sensitivity = 0.15 

    # Factor de ajuste: >1 si jugó un calendario más difícil, <1 si fue más fácil.
    # Este factor ajusta la "calidad percibida" del equipo basada en su calendario.
    ajuste_local_raw = 1.0 + (tpg_opp_local - league_avg_tpg) * sos_sensitivity
    ajuste_visitante_raw = 1.0 + (tpg_opp_visitante - league_avg_tpg) * sos_sensitivity
    
    # Limitar los ajustes para evitar valores extremos (ej., +/- 30% como máximo)
    # Un equipo no se vuelve el doble de bueno/malo solo por el calendario.
    ajuste_local = max(0.7, min(ajuste_local_raw, 1.3))
    ajuste_visitante = max(0.7, min(ajuste_visitante_raw, 1.3))

    # --- Paso 4: Incorporar historial entre ellos como sesgo (Bias) ---
    # Esta parte parece razonable como está. Es un ajuste aditivo.
    historial = df_jugados[
        ((df_jugados["Local"] == local) & (df_jugados["Visitante"] == visitante)) |
        ((df_jugados["Local"] == visitante) & (df_jugados["Visitante"] == local)) # Corregido: Visitante == local para el segundo caso
    ].copy() # Añadido .copy() para evitar SettingWithCopyWarning si se modifica historial (no es el caso aquí, pero buena práctica)


    sesgo_local = 0
    sesgo_visitante = 0
    if not historial.empty:
        pf_hist_local_list = []
        pf_hist_visitante_list = []
        for _, row in historial.iterrows():
            # Asumiendo que ResultadoL/V son los scores del partido, no strings con bonus
            pf_l_hist, _ = parse_resultado(row["ResultadoL"])
            pf_v_hist, _ = parse_resultado(row["ResultadoV"])

            if pf_l_hist is not None and pf_v_hist is not None:
                if row["Local"] == local: # 'local' de la predicción fue Local en este partido histórico
                    pf_hist_local_list.append(pf_l_hist)
                    pf_hist_visitante_list.append(pf_v_hist)
                else: # 'local' de la predicción fue Visitante en este partido histórico (o sea, 'visitante' fue Local)
                    pf_hist_local_list.append(pf_v_hist)
                    pf_hist_visitante_list.append(pf_l_hist)
        
        # El sesgo es cuánto más/menos anota el equipo contra ESTE rival en particular,
        # comparado con su promedio general.
        if pf_hist_local_list: # Si hay historial, calcular el promedio de PF del equipo local en esos partidos
            avg_pf_local_vs_visitante = np.mean(pf_hist_local_list)
            sesgo_local = avg_pf_local_vs_visitante - pf_local
        
        if pf_hist_visitante_list: # Si hay historial, calcular el promedio de PF del equipo visitante en esos partidos
            avg_pf_visitante_vs_local = np.mean(pf_hist_visitante_list)
            sesgo_visitante = avg_pf_visitante_vs_local - pf_visitante


    # --- Paso 5: Predicción Final (Aplicando Ajustes Refinados) ---

    # PF ajustado: si ajuste > 1 (calendario difícil), el equipo es "mejor" de lo que parece, PF sube.
    # PC ajustado: si ajuste > 1 (calendario difícil), el equipo es "mejor", PC baja (concede menos).
    
    adj_pf_local = pf_local * ajuste_local
    # Si ajuste_local es 0 (evitado por el cap), pc_local se mantendría. El cap [0.7, 1.3] lo previene.
    adj_pc_local = pc_local / ajuste_local 

    adj_pf_visitante = pf_visitante * ajuste_visitante
    adj_pc_visitante = pc_visitante / ajuste_visitante

    # Predicción base: Promedio del ataque ajustado de un equipo y la defensa ajustada del otro.
    pred_local_base = (adj_pf_local + adj_pc_visitante) / 2
    pred_visitante_base = (adj_pf_visitante + adj_pc_local) / 2

    # Aplicar el sesgo del historial directo
    pred_l_final = round(pred_local_base + sesgo_local)
    pred_v_final = round(pred_visitante_base + sesgo_visitante)

    # No permitir negativos y asegurar que sean enteros
    return max(0, int(pred_l_final)), max(0, int(pred_v_final))


st.set_page_config(page_title="Rugby Juveniles", layout="wide")

# Estilo para centrar el contenido y limitar el ancho al 80%
st.markdown("""
    <style>
        .main-container {
            max-width: 80%;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
    </style>
    <div class="main-container">
""", unsafe_allow_html=True)

st.title("🏉 Tabla de Posiciones - Rugby Juveniles")


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

st.markdown("### 📅 Seleccioná el Año de Nacimiento:")

# --- Inicio del bloque modificado ---

# Asegurarse de que default_year esté en las opciones para calcular el índice correcto
# y que todas las opciones sean strings para st.radio
options_str = [str(opt) for opt in options_for_selectbox]
default_year_str = str(default_year)

try:
    default_index = options_str.index(default_year_str)
except ValueError:
    default_index = 0 # Si default_year no está en la lista, selecciona el primero
    if not options_str: # Si no hay opciones, no se puede seleccionar nada
        st.error("No hay años disponibles para seleccionar.")
        st.stop() # Detener la ejecución si no hay opciones

# Usar st.radio para la selección
# El callback actualiza st.session_state inmediatamente cuando cambia la selección
def update_selected_year():
    st.session_state["ano_nac_seleccionado_str"] = st.session_state["radio_year_selector"]

selected_year_from_radio = st.radio(
    label="Selecciona el año:",  # El label se puede ocultar si se desea
    options=options_str,
    index=default_index,
    horizontal=True,
    key="radio_year_selector", # Clave única para el widget
    on_change=update_selected_year, # Actualiza el session_state cuando cambia
    label_visibility="collapsed" # Oculta el label "Selecciona el año:"
)

# Establecer el valor inicial en session_state si aún no existe o si el radio lo actualizó
if "ano_nac_seleccionado_str" not in st.session_state:
    st.session_state["ano_nac_seleccionado_str"] = selected_year_from_radio
elif st.session_state["radio_year_selector"] != st.session_state.get("ano_nac_seleccionado_str"):
    # Esto es por si el estado del radio y el session_state se desincronizan
    # (aunque on_change debería mantenerlos sincronizados)
    st.session_state["ano_nac_seleccionado_str"] = st.session_state["radio_year_selector"]


# Obtener el valor seleccionado de session_state para el resto de tu lógica
ano_nac_seleccionado_str = st.session_state.get("ano_nac_seleccionado_str", default_year_str if options_str else None)

# --- Fin del bloque modificado ---

# Reinicializar clubes_checklist si cambió la división
if "estado_anno_anterior" not in st.session_state or st.session_state["estado_anno_anterior"] != ano_nac_seleccionado_str:
    st.session_state["clubes_checklist"] = {}
    st.session_state["estado_anno_anterior"] = ano_nac_seleccionado_str

st.markdown("---")

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
            # Texto instructivo (puedes usarlo junto con el CSS)
            st.caption("💡 Navega entre las diferentes secciones haciendo clic en los títulos de abajo...")
          
            # Creamos las pestañas
            tab1, tab3, tab2, tab5, tab4 = st.tabs([
                "📊 Tabla de Posiciones", 
                "📈 Análisis por equipo", 
                "📅 Partidos Pendientes", 
                "🟦 Tabla de Tarjetas",
                "🔮 La bola de cristal..."
            ])
            # --- TABLA DE POSICIONES ---
            with tab1:
                df_jugados = df_raw_data[df_raw_data["Estado"].str.startswith("Cerrado")].copy()
                st.subheader("📊 Tabla de posiciones")
                st.markdown("&nbsp;") # This adds a small, non-breaking space
                tabla_posiciones = procesar_partidos(df_jugados)

                # Estilizar la tabla (Fondo verde para el TOP 4 de clasificación)
                def color_clasificacion(row):
                    if row.name < 4:  # Índices 0, 1, 2, 3 correspondientes al Top 4
                        return ['background-color: rgba(46, 204, 113, 0.15)'] * len(row)
                    return [''] * len(row)

                tabla_estilizada = tabla_posiciones.style.apply(color_clasificacion, axis=1)

                # Calcular altura dinámica
                filas = len(tabla_posiciones)
                altura = int(filas * 35 + 40)

                st.dataframe(tabla_estilizada, hide_index=True, use_container_width=True, height=altura)

            # --- PARTIDOS PENDIENTES ---
            with tab2:
                st.subheader("📅 Partidos pendientes")

                df_pendientes = df_raw_data[df_raw_data["Estado"] == "Pendiente"].copy()
                df_pendientes["Fecha_dt"] = pd.to_datetime(df_pendientes["Fecha y Hora"], errors="coerce", dayfirst=True)
                df_pendientes["Fecha"] = df_pendientes["Fecha_dt"].dt.strftime('%d/%m/%Y %H:%M')
                clubes_locales = df_pendientes["Local"].unique()
                clubes_visitantes = df_pendientes["Visitante"].unique()
                clubes_unicos = sorted(set(clubes_locales) | set(clubes_visitantes))

                # 1. Filtro Inteligente Nativo
                clubes_seleccionados = st.multiselect(
                    "🔎 Filtrar por clubes (dejá vacío para ver todos):", 
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
                    st.metric("🏉 Partidos Pendientes Totales", len(df_filtrado))
                with col_metric_2:
                    if not df_filtrado.empty and not df_filtrado["Fecha_dt"].dropna().empty:
                        prox_fecha = df_filtrado["Fecha_dt"].dropna().min()
                        prox_partido_str = prox_fecha.strftime('%d/%m/%Y %H:%M')
                    else:
                        prox_partido_str = "N/A"
                    st.metric("📆 Próximo Partido", prox_partido_str)

                # 3. Mejoras Visuales en el DataFrame
                dias_espanol = {
                    "Monday": "Lunes", 
                    "Tuesday": "Martes", 
                    "Wednesday": "Miércoles", 
                    "Thursday": "Jueves", 
                    "Friday": "Viernes", 
                    "Saturday": "Sábado", 
                    "Sunday": "Domingo"
                }

                if not df_filtrado.empty:
                    df_filtrado["Día"] = df_filtrado["Fecha_dt"].dt.day_name().map(dias_espanol).fillna("-")
                    df_mostrar = df_filtrado.rename(columns={
                        "Local": "🏠 Local", 
                        "Visitante": "✈️ Visitante"
                    })
                    columnas_mostrar = ["Día", "Fecha", "🏠 Local", "✈️ Visitante"]
                else:
                    df_mostrar = pd.DataFrame(columns=["Día", "Fecha", "🏠 Local", "✈️ Visitante"])
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
                st.subheader("📈 Análisis por equipo")

                equipos = sorted(tabla_posiciones["Equipo"].unique())
                # Seleccionar "UNIVERSITARIO" por defecto si existe, si no el primero.
                default_index = equipos.index("UNIVERSITARIO") if "UNIVERSITARIO" in equipos else 0
                equipo_sel = st.selectbox("Seleccioná un equipo", equipos, index=default_index)

                if equipo_sel:
                    # Partidos jugados por el equipo
                    jugados_equipo = df_jugados[(df_jugados["Local"] == equipo_sel) | (df_jugados["Visitante"] == equipo_sel)].copy()
                    
                    if not jugados_equipo.empty: # Verificar si el equipo tiene partidos
                        # Parseo de resultados
                        jugados_equipo["Puntos Equipo"] = jugados_equipo.apply(
                            lambda row: parse_resultado(row["ResultadoL"] if row["Local"] == equipo_sel else row["ResultadoV"])[0],
                            axis=1
                        )
                        jugados_equipo["Puntos Rival"] = jugados_equipo.apply(
                            lambda row: parse_resultado(row["ResultadoV"] if row["Local"] == equipo_sel else row["ResultadoL"])[0],
                            axis=1
                        )
                        jugados_equipo["Pts Torneo Equipo"] = jugados_equipo.apply(
                            lambda row: parse_resultado(row["ResultadoL"] if row["Local"] == equipo_sel else row["ResultadoV"])[1],
                            axis=1
                        )

                        # Cálculo de victorias, empates, derrotas
                        victorias = (jugados_equipo["Puntos Equipo"] > jugados_equipo["Puntos Rival"]).sum()
                        empates = (jugados_equipo["Puntos Equipo"] == jugados_equipo["Puntos Rival"]).sum()
                        derrotas = (jugados_equipo["Puntos Equipo"] < jugados_equipo["Puntos Rival"]).sum()
                        total = len(jugados_equipo)
                        
                        puntos_totales_torneo_equipo = jugados_equipo["Pts Torneo Equipo"].sum()
                        
                        prom_puntos_favor = jugados_equipo['Puntos Equipo'].mean() if total > 0 else 0
                        prom_puntos_contra = jugados_equipo['Puntos Rival'].mean() if total > 0 else 0
                        dif_prom_partido = prom_puntos_favor - prom_puntos_contra

                        # Dividir en dos columnas principales: una para datos y otra para el gráfico
                        col_datos, col_grafico = st.columns([0.6, 0.4]) 

                        with col_datos:
                            st.markdown(f"#### Estadísticas de {equipo_sel}")
                            
                            # Crear dos sub-columnas dentro de col_datos para las métricas
                            sub_col_datos1, sub_col_datos2 = st.columns(2)

                            with sub_col_datos1:
                                st.metric("🏉 Partidos jugados", total)
                                st.metric("🔢 Pts. torneo (bonus)", puntos_totales_torneo_equipo)
                                st.metric("📊 Prom. P. Favor / Contra", f"{prom_puntos_favor:.1f} / {prom_puntos_contra:.1f}")
                                # Nueva métrica con color dinámico usando delta de Streamlit
                                st.metric("⚖️ Dif. Promedio por Partido", f"{dif_prom_partido:+.1f}", delta=f"{dif_prom_partido:+.1f}")
                            
                            with sub_col_datos2:
                                st.metric("✅ Ganados", f"{victorias} ({victorias/total:.0%})" if total > 0 else "0 (0%)")
                                st.metric("➖ Empatados", f"{empates} ({empates/total:.0%})" if total > 0 else "0 (0%)")
                                st.metric("❌ Perdidos", f"{derrotas} ({derrotas/total:.0%})" if total > 0 else "0 (0%)")

                        with col_grafico:
                            st.markdown("#### Distribución de Resultados")
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
                                    margin=dict(t=20, b=20, l=20, r=20)
                                )
                                st.plotly_chart(fig1, use_container_width=True)
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
                        xaxis_title="Fecha y Rival",
                        yaxis_title="Puntos",
                        legend_title="Tipo",
                        hovermode="x unified",
                        margin=dict(t=50, b=40, l=40, r=20)
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                    # Definir siempre esta variable, esté o no en partidos pendientes
                    equipo_seleccionado = equipo_sel
                    # ---------- Mostrar partidos pendientes ----------
                    pendientes_equipo = df_pendientes[
                        (df_pendientes["Local"] == equipo_sel) | (df_pendientes["Visitante"] == equipo_sel)
                    ].copy()

                    if not pendientes_equipo.empty:
                        st.subheader("📅 Partidos pendientes del equipo")

                        equipo_seleccionado = equipo_sel  # Asegurate de que esta variable tenga el nombre del equipo

                        # Convertir y formatear la fecha sin segundos
                        pendientes_equipo["Fecha_dt"] = pd.to_datetime(pendientes_equipo["Fecha y Hora"], errors="coerce", dayfirst=True)
                        pendientes_equipo["Fecha"] = pendientes_equipo["Fecha_dt"].dt.strftime('%d/%m/%Y %H:%M')

                        # Determinar rival y condición
                        pendientes_equipo["Rival"] = pendientes_equipo.apply(
                            lambda row: row["Visitante"] if row["Local"] == equipo_seleccionado else row["Local"],
                            axis=1
                        )
                        # Cambio visual de emojis para identificar Local / Visitante
                        pendientes_equipo["Condición"] = pendientes_equipo["Local"].apply(
                            lambda x: "🏠 Local" if x == equipo_seleccionado else "✈️ Visitante"
                        )

                        # Ordenar por fecha_dt para que sea cronológico
                        pendientes_equipo = pendientes_equipo.sort_values("Fecha_dt")

                        # Nombre dinámico para la columna
                        columna_condicion = f"{equipo_seleccionado} juega como"

                        # Renombrar la columna para que sea más clara
                        pendientes_equipo.rename(columns={"Condición": columna_condicion}, inplace=True)

                        # Mostrar tabla
                        st.dataframe(
                            pendientes_equipo[["Rival", columna_condicion, "Fecha"]],
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("Este equipo no tiene partidos pendientes.")

                    # Cargar tarjetas si hay hoja disponible
                    df_tarjetas = get_tarjetas_data(gs_client, ano_nac_seleccionado_str)

                    # Filtrar tarjetas del equipo seleccionado
                    df_tarjetas_equipo = df_tarjetas[df_tarjetas["Equipo"] == equipo_seleccionado]

                    if not df_tarjetas_equipo.empty:
                        st.subheader("🟨🟥 Incidencias disciplinarias")
                        
                        # Mostrar resumen rápido
                        total_amarillas = df_tarjetas_equipo["Incidencia"].str.contains("amarilla", case=False).sum()
                        total_rojas = df_tarjetas_equipo["Incidencia"].str.contains("roja", case=False).sum()
                        total_azules = df_tarjetas_equipo["Incidencia"].str.contains("azul", case=False).sum()

                        st.markdown(f"""
                        - 🟨 **Total amarillas:** {total_amarillas}  
                        - 🟥 **Total rojas:** {total_rojas}  
                        - 🔵 **Total azules:** {total_azules}  
                        """)

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
                    st.subheader("🔮 La bola de cristal... puede fallar! (y lo va a hacer!)")

                    if df_pendientes.empty:
                        st.info("No hay partidos pendientes para mostrar predicciones.")
                    elif df_jugados.empty:
                        st.warning("No hay datos de partidos jugados (cerrados). No se pueden generar predicciones detalladas.")
                        st.dataframe(df_pendientes[["Local", "Visitante", "Fecha y Hora"]], hide_index=True, use_container_width=True)
                    else:
                        st.markdown("**🔍 Filtrado de partidos para predicción**")

                        clubes_locales = df_pendientes["Local"].unique()
                        clubes_visitantes = df_pendientes["Visitante"].unique()
                        clubes_unicos = sorted(set(clubes_locales) | set(clubes_visitantes))

                        # Inicializar estado de selección si no existe
                        if "clubes_checklist_pred" not in st.session_state:
                            st.session_state["clubes_checklist_pred"] = {club: False for club in clubes_unicos}

                        if "seleccion_pred_todos" not in st.session_state:
                            st.session_state["seleccion_pred_todos"] = False  # default: ninguno seleccionado

                        with st.expander("✅ Elegí los clubes para predecir (ninguno seleccionado por defecto)"):
                            col_button, _ = st.columns([1, 3])
                            accion = "Seleccionar todos" if not all(st.session_state["clubes_checklist_pred"].values()) else "Deseleccionar todos"
                            if col_button.button(accion, key="boton_toggle_pred"):
                                nuevo_estado = not all(st.session_state["clubes_checklist_pred"].values())
                                for club in clubes_unicos:
                                    st.session_state["clubes_checklist_pred"][club] = nuevo_estado

                        # Inicializar si no existe
                        if "clubes_checklist_pred" not in st.session_state:
                            st.session_state["clubes_checklist_pred"] = {}

                        # Asegurarse de que todos los clubes estén presentes en el diccionario
                        for club in clubes_unicos:
                            if club not in st.session_state["clubes_checklist_pred"]:
                                st.session_state["clubes_checklist_pred"][club] = True

                        # Mostrar los checkboxes
                        cols = st.columns(3)
                        for i, club in enumerate(clubes_unicos):
                            col = cols[i % 3]
                            st.session_state["clubes_checklist_pred"][club] = col.checkbox(
                                club,
                                value=st.session_state["clubes_checklist_pred"][club],
                                key=f"check_pred_{club}"
                            )

                        # Filtrado activo
                        clubes_pred_activos = [
                            club for club, activo in st.session_state["clubes_checklist_pred"].items() if activo
                        ]

                        if not clubes_pred_activos:
                            st.info("Seleccioná uno o más clubes para mostrar las predicciones.")
                        else:
                            df_pendientes_filtrados = df_pendientes[
                                df_pendientes["Local"].isin(clubes_pred_activos) |
                                df_pendientes["Visitante"].isin(clubes_pred_activos)
                            ].sort_values("Fecha")

                            predicciones_list = []
                            for _, row in df_pendientes_filtrados.iterrows():
                                local = row["Local"]
                                visitante = row["Visitante"]
                                pred_l, pred_v = predecir_resultado(local, visitante, tabla_posiciones, df_jugados, parse_resultado)
                                predicciones_list.append({
                                    "Fecha y Hora": row["Fecha y Hora"],
                                    "Local": local,
                                    "Pred. Local": pred_l if pred_l is not None else "N/A",
                                    "Visitante": visitante,
                                    "Pred. Visitante": pred_v if pred_v is not None else "N/A"
                                })

                            df_pred = pd.DataFrame(predicciones_list)

                            columnas_esperadas = ["Fecha y Hora", "Local", "Pred. Local", "Visitante", "Pred. Visitante"]
                            
                            if not df_pred.empty and all(col in df_pred.columns for col in columnas_esperadas):
                                st.markdown(f"**Mostrando predicciones para:** {', '.join(clubes_pred_activos)}")
                                st.dataframe(
                                    df_pred[columnas_esperadas],
                                    hide_index=True,
                                    use_container_width=True
                                )
                            else:
                                st.info("No hay predicciones disponibles para mostrar.")

            # ---------- TABLA DE TARJETAS ----------
            with tab5:
                df_tarjetas = get_tarjetas_data(gs_client, ano_nac_seleccionado_str)

                if df_tarjetas.empty:
                    st.info("No hay registros de tarjetas para esta división.")
                else:
                    st.subheader("🟦 Tabla de Disciplina")

                    # --- 1. Procesamiento de Datos ---
                    df_tarjetas["Incidencia"] = df_tarjetas["Incidencia"].str.lower().str.strip()
                    
                    # Conteo total para KPIs de toda la división
                    total_y = df_tarjetas[df_tarjetas["Incidencia"].str.contains("amarilla")].shape[0]
                    total_r = df_tarjetas[df_tarjetas["Incidencia"].str.contains("roja")].shape[0]
                    total_b = df_tarjetas[df_tarjetas["Incidencia"].str.contains("azul")].shape[0]

                    # --- 2. Panel de Métricas ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🟨 Amarillas Totales", total_y)
                    m2.metric("🟥 Rojas Totales", total_r)
                    m3.metric("🔵 Azules Totales", total_b)

                    # --- 3. Filtro por Club ---
                    clubes_con_tarjetas = sorted(df_tarjetas["Equipo"].unique())
                    clubes_sel = st.multiselect(
                        "🔎 Filtrar por club (dejar vacío para ver todos):", 
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
                        "amarilla": "🟨 Amarillas", 
                        "roja": "🟥 Rojas", 
                        "azul": "🔵 Azules"
                    }
                    
                    for col_original, col_nueva in mapeo_columnas.items():
                        if col_original not in resumen.columns:
                            resumen[col_original] = 0
                        resumen = resumen.rename(columns={col_original: col_nueva})

                    # Ordenar por gravedad (Rojas valen más puntos en un ranking inverso de disciplina)
                    resumen["_score"] = (resumen["🟨 Amarillas"] * 1) + (resumen["🟥 Rojas"] * 5) + (resumen["🔵 Azules"] * 2)
                    resumen = resumen.sort_values("_score", ascending=False).drop(columns=["_score"])

                    # --- 5. Gráfico de Conducta (Plotly) ---
                    st.markdown("#### Conducta por Club")
                    if not resumen.empty:
                        # Preparar datos para Plotly format largo
                        columnas_cards = ["🟨 Amarillas", "🔵 Azules", "🟥 Rojas"]
                        df_plot = resumen.melt(id_vars="Equipo", value_vars=columnas_cards, var_name="Tipo", value_name="Cantidad")
                        
                        fig_cards = px.bar(
                            df_plot, 
                            x="Cantidad", 
                            y="Equipo", 
                            color="Tipo",
                            orientation='h',
                            color_discrete_map={
                                "🟨 Amarillas": "#f1c40f", 
                                "🟥 Rojas": "#e74c3c", 
                                "🔵 Azules": "#3498db"
                            },
                        )
                        fig_cards.update_layout(
                            barmode='stack', 
                            yaxis={'categoryorder':'total ascending'},
                            height=max(min(len(resumen) * 35 + 100, 600), 300),
                            margin=dict(l=20, r=20, t=20, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_cards, use_container_width=True)

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
