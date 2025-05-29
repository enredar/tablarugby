import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re



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



# ---------- Carga de datos ----------
uploaded_file = st.file_uploader("📁 Subí el archivo CSV con los resultados", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    # Validar columnas requeridas
    expected_cols = ['Nro.', 'Local', 'ResultadoL', 'ResultadoV', 'Visitante', 'Fecha y Hora', 'Estado']
    if not all(col in df_raw.columns for col in expected_cols):
        st.dataframe(df_raw)
        st.error("El archivo no tiene las columnas esperadas")
    else:
        # Tabla de posiciones
        df_jugados = df_raw[df_raw["Estado"].str.startswith("Cerrado")].copy()
        st.subheader("📊 Tabla de posiciones")
        tabla_posiciones = procesar_partidos(df_jugados)
        st.dataframe(tabla_posiciones, hide_index=True, use_container_width=True)

        # Partidos pendientes
        st.markdown("---")
        st.subheader("📅 Partidos pendientes")

        df_pendientes = df_raw[df_raw["Estado"] == "Pendiente"].copy()
        df_pendientes["Fecha"] = pd.to_datetime(df_pendientes["Fecha y Hora"], errors="coerce")

        col1, col2 = st.columns(2)
        with col1:
            ordenar_por = st.selectbox("Ordenar por", ["Fecha", "Local", "Visitante"])
        with col2:
            asc = st.radio("Orden", ["Ascendente", "Descendente"], horizontal=True) == "Ascendente"

        if ordenar_por == "Fecha":
            df_pendientes = df_pendientes.sort_values("Fecha", ascending=asc)
        else:
            df_pendientes = df_pendientes.sort_values(ordenar_por, ascending=asc)

        st.dataframe(df_pendientes[["Local", "Visitante", "Fecha y Hora"]], hide_index=True, use_container_width=True)

        # ---------- Análisis por equipo ----------
        st.markdown("---")
        st.subheader("📈 Análisis por equipo")

        equipos = sorted(tabla_posiciones["Equipo"].unique())
        equipo_sel = st.selectbox("Seleccioná un equipo", equipos)

        if equipo_sel:
            # Partidos jugados por el equipo
            jugados_equipo = df_jugados[(df_jugados["Local"] == equipo_sel) | (df_jugados["Visitante"] == equipo_sel)].copy()
            
            # Parseo de resultados
            jugados_equipo["Puntos Equipo"] = jugados_equipo.apply(
                lambda row: parse_resultado(row["ResultadoL"] if row["Local"] == equipo_sel else row["ResultadoV"])[0],
                axis=1
            )
            jugados_equipo["Puntos Rival"] = jugados_equipo.apply(
                lambda row: parse_resultado(row["ResultadoV"] if row["Local"] == equipo_sel else row["ResultadoL"])[0],
                axis=1
            )
            jugados_equipo["Pts Torneo"] = jugados_equipo.apply(
                lambda row: parse_resultado(row["ResultadoL"] if row["Local"] == equipo_sel else row["ResultadoV"])[1],
                axis=1
            )

            # Cálculo de victorias, empates, derrotas
            victorias = (jugados_equipo["Puntos Equipo"] > jugados_equipo["Puntos Rival"]).sum()
            empates = (jugados_equipo["Puntos Equipo"] == jugados_equipo["Puntos Rival"]).sum()
            derrotas = (jugados_equipo["Puntos Equipo"] < jugados_equipo["Puntos Rival"]).sum()
            total = len(jugados_equipo)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏉 Partidos jugados", total)
                st.metric("✅ Ganados", f"{victorias} ({victorias/total:.0%})")
            with col2:
                st.metric("🔢 Puntos en el torneo", jugados_equipo["Pts Torneo"].sum())
                st.metric("❌ Perdidos", f"{derrotas} ({derrotas/total:.0%})")
            with col3:
                st.metric("📊 Prom. P. a Favor / P. en Contra", f"{jugados_equipo['Puntos Equipo'].mean():.1f} / {jugados_equipo['Puntos Rival'].mean():.1f}")
                st.metric("➖ Empatados", f"{empates} ({empates/total:.0%})")

            # # ---------- Gráfico de resultados de barras----------
            # fig1, ax1 = plt.subplots(figsize=(3, 3))  # más compacto
            # ax1.pie(
            #     [victorias, empates, derrotas],
            #     labels=["Ganados", "Empatados", "Perdidos"],
            #     autopct="%1.1f%%",
            #     startangle=90,
            #     colors=["#2ecc71", "#f1c40f", "#e74c3c"]
            # )
            # ax1.axis("equal")
            # st.pyplot(fig1)





            # ---------- Evolución de puntos ----------
            jugados_equipo["Fecha"] = pd.to_datetime(jugados_equipo["Fecha y Hora"], errors="coerce")
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
            # ---------- Mostrar partidos pendientes ----------
            pendientes_equipo = df_pendientes[
                (df_pendientes["Local"] == equipo_sel) | (df_pendientes["Visitante"] == equipo_sel)
            ].copy()

            if not pendientes_equipo.empty:
                st.subheader("📅 Partidos pendientes del equipo")
                pendientes_equipo["Fecha"] = pd.to_datetime(pendientes_equipo["Fecha y Hora"], errors="coerce")
                pendientes_equipo = pendientes_equipo.sort_values("Fecha")
                st.dataframe(pendientes_equipo[["Local", "Visitante", "Fecha y Hora"]], hide_index=True, use_container_width=True)
            else:
                st.info("Este equipo no tiene partidos pendientes.")

            # ---------- Predicción de partidos pendientes ----------
            st.markdown("---")
            st.subheader("🔮 Predicción de partidos pendientes")

            if df_pendientes.empty:
                st.info("No hay partidos pendientes para mostrar predicciones.")
            elif df_jugados.empty:
                st.warning("No hay datos de partidos jugados (cerrados). No se pueden generar predicciones detalladas.")
                # Opcionalmente, mostrar solo los partidos pendientes sin predicción:
                st.dataframe(df_pendientes[["Local", "Visitante", "Fecha y Hora"]], hide_index=True, use_container_width=True)
            # Omití la comprobación de tabla_posiciones.empty aquí porque predecir_resultado tiene fallbacks internos
            # si un rival no está en tabla_posiciones o tabla_posiciones está vacía (fuerza_rivales = 1.0)
            else:
                predicciones_list = [] # Renombrado para evitar conflicto si predicciones es una función
                for _, row in df_pendientes.iterrows():
                    local = row["Local"]
                    visitante = row["Visitante"]
                    
                    # Es buena idea verificar si los equipos existen en tabla_posiciones si la lógica de fuerza depende de ello
                    # aunque tu función actual busca rivales y tiene un default.

                    pred_l, pred_v = predecir_resultado(local, visitante, tabla_posiciones, df_jugados, parse_resultado)
                    
                    predicciones_list.append({
                        "Local": local,
                        "Visitante": visitante,
                        "Pred. Local": pred_l if pred_l is not None else "N/A", # Muestra N/A si es None
                        "Pred. Visitante": pred_v if pred_v is not None else "N/A", # Muestra N/A si es None
                        "Fecha y Hora": row["Fecha y Hora"]
                    })

                if predicciones_list:
                    df_pred = pd.DataFrame(predicciones_list)
                    st.dataframe(df_pred[["Fecha y Hora", "Local", "Pred. Local", "Visitante", "Pred. Visitante"]], hide_index=True, use_container_width=True)
                elif not df_pendientes.empty: # Había partidos pendientes pero no se generó la lista (improbable con el N/A)
                    st.info("No se pudieron generar predicciones para los partidos pendientes.")


else:
    st.info("Esperando que cargues un archivo CSV con los resultados.")


# Cerramos el div que abrimos antes
st.markdown("</div>", unsafe_allow_html=True)