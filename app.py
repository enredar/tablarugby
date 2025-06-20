import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from google_sheets_client import get_gspread_client, get_division_data, get_available_birth_years, get_tarjetas_data

# ---------- Funciones auxiliares ----------

def parse_resultado(resultado):
    """
    Extrae los puntos del partido y los puntos para la tabla.
    Ej: "25 [4]" → (25, 4)
    """
    match = re.match(r"(\d+)\s*\[(\d+)\]", str(resultado).strip())
    if match:
        return int(match.group(1)), int(match.group(2))
    elif resultado == '-' or resultado.strip() == '':
        return None, None
    else:
        return int(resultado), 0  # caso raro sin puntos campeonato

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



    # Calcular diferencia
    for equipo in posiciones:
        posiciones[equipo]["DIF"] = posiciones[equipo]["PF"] - posiciones[equipo]["PC"]

    tabla = pd.DataFrame(list(posiciones.values()))
    tabla = tabla.sort_values(by=["PTS", "DIF", "PF"], ascending=False).reset_index(drop=True)
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

# --- Selección de División (Año de Nacimiento) ---
# Obtener dinámicamente los años/pestañas disponibles
available_years_str = get_available_birth_years(gs_client)
# Convertir a int para ordenar si son numéricos, luego a str para el selectbox
try:
    available_years_int = sorted([int(y) for y in available_years_str if y.isdigit()], reverse=True)
    options_for_selectbox = [str(y) for y in available_years_int]
except ValueError:
    options_for_selectbox = sorted(available_years_str) # Si no son todos números, orden alfabético

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

                # Calcular altura dinámica (35 px por fila + 40 de margen por encabezado)
                filas = len(tabla_posiciones)
                altura = int(filas * 35 + 40)

                st.dataframe(tabla_posiciones, hide_index=True, use_container_width=True, height=altura)

            # --- PARTIDOS PENDIENTES ---
            with tab2:
                st.subheader("📅 Partidos pendientes")

                df_pendientes = df_raw_data[df_raw_data["Estado"] == "Pendiente"].copy()
                df_pendientes["Fecha"] = pd.to_datetime(df_pendientes["Fecha y Hora"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y %H:%M')
                clubes_locales = df_pendientes["Local"].unique()
                clubes_visitantes = df_pendientes["Visitante"].unique()
                clubes_unicos = sorted(set(clubes_locales) | set(clubes_visitantes))

                modo_filtro = st.selectbox("🔎 Filtrar partidos pendientes por clubes:", ["Todos los clubes", "Seleccionar clubes..."])

                clubes_activos = clubes_unicos.copy()

                if modo_filtro == "Seleccionar clubes...":
                    if "clubes_checklist" not in st.session_state:
                        st.session_state["clubes_checklist"] = {club: True for club in clubes_unicos}

                    with st.expander("✅ Elegí los clubes que querés ver"):
                        cols = st.columns(3)
                        for i, club in enumerate(clubes_unicos):
                            col = cols[i % 3]
                            st.session_state["clubes_checklist"][club] = col.checkbox(
                                club,
                                value=st.session_state["clubes_checklist"][club],
                                key=f"check_{club}"
                            )

                    clubes_activos = [club for club, activo in st.session_state["clubes_checklist"].items() if activo]

                    if len(clubes_activos) == 0:
                        st.warning("No seleccionaste ningún club. No se mostrarán partidos.")
                    else:
                        st.markdown(f"**Mostrando partidos de:** {', '.join(clubes_activos)}")

                df_filtrado = df_pendientes[
                    df_pendientes["Local"].isin(clubes_activos) |
                    df_pendientes["Visitante"].isin(clubes_activos)
                ].sort_values("Fecha")

                # Calcular altura dinámica (35 px por fila + 40 de margen por encabezado)
                filas = len(df_filtrado)
                altura = min(int(filas * 35 + 40), 800)

                st.dataframe(
                    df_filtrado[["Local", "Visitante", "Fecha"]],
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
                            
                            with sub_col_datos2:
                                st.metric("✅ Ganados", f"{victorias} ({victorias/total:.0%})" if total > 0 else "0 (0%)")
                                st.metric("➖ Empatados", f"{empates} ({empates/total:.0%})" if total > 0 else "0 (0%)")
                                st.metric("❌ Perdidos", f"{derrotas} ({derrotas/total:.0%})" if total > 0 else "0 (0%)")

                        with col_grafico:
                            st.markdown("#### Distribución de Resultados")
                            if total > 0: 
                                fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
                                ax1.pie(
                                    [victorias, empates, derrotas],
                                    labels=["Ganados", "Empatados", "Perdidos"],
                                    autopct="%1.1f%%",
                                    startangle=90,
                                    colors=["#2ecc71", "#f1c40f", "#e74c3c"],
                                    wedgeprops={'edgecolor': 'white'} 
                                )
                                ax1.axis("equal") 
                                st.pyplot(fig1)
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

                    # Crear DataFrame para gráfico de barras agrupadas
                    df_barras = pd.melt(
                        jugados_equipo[["Etiqueta", "Puntos Equipo", "Puntos Rival"]],
                        id_vars="Etiqueta",
                        var_name="Tipo",
                        value_name="Puntos"
                    )

                    # Renombrar los tipos
                    df_barras["Tipo"] = df_barras["Tipo"].map({
                        "Puntos Equipo": "A favor",
                        "Puntos Rival": "En contra"
                    })

                    # Gráfico de barras
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    barplot = sns.barplot(data=df_barras, x="Etiqueta", y="Puntos", hue="Tipo", ax=ax2, palette=["#3498db", "#e74c3c"])

                    # Agregar los valores encima de las barras
                    for container in ax2.containers:
                        ax2.bar_label(container, fmt="%.0f", label_type='edge', padding=3, fontsize=8)

                    # Mejoras estéticas del eje X
                    ax2.set_title(f"Puntos por partido - {equipo_sel}")
                    ax2.set_ylabel("Puntos")
                    ax2.set_xlabel("Rival (Fecha)")
                    ax2.tick_params(axis='x', rotation=60, labelsize=8)
                    fig2.tight_layout()

                    st.pyplot(fig2)

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
                        pendientes_equipo["Fecha"] = pd.to_datetime(pendientes_equipo["Fecha y Hora"], errors="coerce", dayfirst=True)
                        pendientes_equipo["Fecha"] = pendientes_equipo["Fecha"].dt.strftime('%d/%m/%Y %H:%M')

                        # Determinar rival y condición
                        pendientes_equipo["Rival"] = pendientes_equipo.apply(
                            lambda row: row["Visitante"] if row["Local"] == equipo_seleccionado else row["Local"],
                            axis=1
                        )
                        pendientes_equipo["Condición"] = pendientes_equipo["Local"].apply(
                            lambda x: "Local" if x == equipo_seleccionado else "Visitante"
                        )

                        # Ordenar por fecha
                        pendientes_equipo = pendientes_equipo.sort_values("Fecha")

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
                    st.subheader("🟨🟥🔵 Tarjetas por equipo")

                    # Normalizamos las incidencias
                    df_tarjetas["Incidencia"] = df_tarjetas["Incidencia"].str.lower()

                    # Agrupamos y convertimos a tabla
                    resumen = df_tarjetas.groupby("Equipo")["Incidencia"].value_counts().unstack(fill_value=0).reset_index()

                    # Aseguramos que existan las columnas esperadas, incluso si no hay tarjetas de ese tipo
                    for col in ["amarilla", "roja", "azul"]:
                        if col not in resumen.columns:
                            resumen[col] = 0

                    # Renombramos para visualización
                    resumen = resumen.rename(columns={
                        "amarilla": "🟨 Amarillas",
                        "roja": "🟥 Rojas",
                        "azul": "🔵 Azules"
                    })

                    # Agregamos total
                    resumen["🧮 Total"] = resumen[["🟨 Amarillas", "🟥 Rojas", "🔵 Azules"]].sum(axis=1)

                    # Ordenamos
                    resumen = resumen.sort_values(by="🧮 Total", ascending=False)

                    st.dataframe(
                        resumen,
                        use_container_width=True,
                        hide_index=True,
                        height=min(len(resumen) * 35 + 40, 600)
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
