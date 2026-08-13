"""Panel de administración: login + carga de datos (pegado masivo).

Se muestra en la sidebar de la app cuando SUPABASE_URL está configurado.
El cliente de lectura usa la anon key; las escrituras usan el token del
admin logueado (RLS exige claim role=admin).
"""

import re

import pandas as pd
import streamlit as st
from supabase import create_client, Client

from logic import parse_resultado


# ---------- Auth ----------

def _base_client() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def is_admin_signed_in() -> bool:
    return bool(st.session_state.get("su_admin_token"))


def sign_out():
    for k in ("su_admin_token", "su_admin_refresh", "su_admin_email"):
        st.session_state.pop(k, None)
    st.rerun()


def _admin_client() -> Client:
    client = _base_client()
    client.auth.set_session(
        st.session_state["su_admin_token"],
        st.session_state["su_admin_refresh"],
    )
    return client


def _render_login():
    st.subheader("Iniciar sesión")
    email = st.text_input("Email", key="su_email")
    password = st.text_input("Contraseña", type="password", key="su_pass")
    if st.button("Entrar", use_container_width=True):
        if not email or not password:
            st.error("Completá email y contraseña.")
            return
        try:
            res = _base_client().auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state["su_admin_token"] = res.session.access_token
            st.session_state["su_admin_refresh"] = res.session.refresh_token
            st.session_state["su_admin_email"] = email
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo iniciar sesión: {e}")


# ---------- Helpers de torneos ----------

def _get_torneos(client) -> pd.DataFrame:
    resp = client.table("torneos").select("*").order("temporada", desc=True).order("id", desc=True).execute()
    return pd.DataFrame(resp.data)


def _get_etapas(client, torneo_id) -> pd.DataFrame:
    resp = client.table("etapas").select("*").eq("torneo_id", torneo_id).order("orden").execute()
    return pd.DataFrame(resp.data)


def _get_or_create_etapa(client, torneo_id, nombre="Regular"):
    resp = client.table("etapas").select("id").eq("torneo_id", torneo_id).eq("nombre", nombre).execute()
    if resp.data:
        client.table("etapas").update({"orden": 0}).eq("id", resp.data[0]["id"]).execute()
        return resp.data[0]["id"]
    res = client.table("etapas").insert({"torneo_id": torneo_id, "nombre": nombre, "orden": 0}).execute()
    return res.data[0]["id"]


def _coerce_str(v):
    """Convierte a string limpio; maneja NaN/None/numéricos."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _get_or_create_equipo(client, torneo_id, nombre):
    nombre = _coerce_str(nombre)
    if not nombre:
        return None
    resp = client.table("equipos").select("id").eq("torneo_id", torneo_id).eq("nombre", nombre).execute()
    if resp.data:
        return resp.data[0]["id"]
    res = client.table("equipos").insert({"torneo_id": torneo_id, "nombre": nombre}).execute()
    return res.data[0]["id"]


# ---------- Parser del pegado masivo ----------

COLUMNAS = ["Nro.", "Local", "ResultadoL", "ResultadoV", "Visitante",
            "Fecha y Hora", "Referee", "Estado"]


def parsear_pegado(texto: str) -> pd.DataFrame:
    """Parsea el texto pegado desde bd.uar (filas separadas por tabs).

    El formato es (heredado directamente de la planilla):
    Nro. | Local | ResultadoL | ResultadoV | Visitante | Fecha y Hora |
    Referee | Estado [| Acciones]
    """
    filas = []
    for linea in texto.splitlines():
        linea = linea.rstrip("\n")
        if not linea.strip():
            continue
        # Saltar fila de encabezado si apareció
        if linea.lstrip().startswith("Nro.") or linea.lstrip().startswith("Nro\t"):
            continue
        celdas = [c.strip() for c in re.split(r"\t+", linea)]
        if len(celdas) < 7:
            continue
        nro = celdas[0]
        local = celdas[1]
        res_local = celdas[2]
        res_visitante = celdas[3]
        visitante = celdas[4]
        fecha_hora = celdas[5]
        referee = celdas[6] if len(celdas) > 6 else ""
        estado = celdas[7] if len(celdas) > 7 else "Pendiente"
        filas.append({
            "Nro.": nro,
            "Local": local,
            "ResultadoL": res_local,
            "ResultadoV": res_visitante,
            "Visitante": visitante,
            "Fecha y Hora": fecha_hora,
            "Referee": referee,
            "Estado": estado,
        })
    return pd.DataFrame(filas, columns=COLUMNAS)


def _estado_normalizado(estado: str) -> str:
    estado = (estado or "").strip()
    if not estado:
        return "Pendiente"
    est = estado.lower()
    if est.startswith("cerrado") or "cerrado" in est:
        return "Cerrado"
    if "en curso" in est or "iniciado" in est:
        return "En Curso"
    return "Pendiente"


def _fecha_parse(fecha_hora):
    try:
        dt = pd.to_datetime(fecha_hora, dayfirst=True)
        return dt if pd.notna(dt) else None
    except Exception:
        return None


# ---------- Carga de partidos (pegado masivo) ----------

def _puede_guardar_tabla(df):
    """Filtra y valida una tabla parseada, devolviendo (rows_validas, errores)."""
    registros = []
    errores = []
    for i, row in df.iterrows():
        res_l, pts_l = parse_resultado(row["ResultadoL"])
        res_v, pts_v = parse_resultado(row["ResultadoV"])

        if res_l is None and res_v is None:
            # Puede ser un partido aún sin cargar resultado pero con estado Pendiente
            registro = {
                "nro": _int_or_none(row["Nro."]),
                "estado": _estado_normalizado(row["Estado"]),
            }
            if registro["estado"] == "Cerrado":
                # Cerrado sin resultado: inconsistencia -> lo reportamos
                errores.append(f"Fila {row['Nro.']}: partido 'Cerrado' sin resultado ({row['Local']} vs {row['Visitante']}).")
                continue
            registros.append(registro)
            continue

        registro = {
            "nro": _int_or_none(row["Nro."]),
            "estado": _estado_normalizado(row["Estado"]),
            "resultado_local": res_l if res_l is not None else 0,
            "resultado_visitante": res_v if res_v is not None else 0,
            "puntos_local": pts_l or (0 if res_l is not None else None),
            "puntos_visitante": pts_v or (0 if res_v is not None else None),
            "fecha_hora": _fecha_parse(row["Fecha y Hora"]),
            "referee": row.get("Referee") or "",
            "local_nombre": row["Local"],
            "visitante_nombre": row["Visitante"],
        }
        registros.append(registro)
    return registros, errores


def _int_or_none(v):
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return None


def _persistir_partidos(client, torneo_id, registros):
    """Inserta o actualiza partidos (upsert por nro). Devuelve (nuevos, actualizados)."""
    etapa_id = _get_or_create_etapa(client, torneo_id)

    # Partidos existentes por nro
    nros = [r["nro"] for r in registros if r.get("nro")]
    existentes = {}
    if nros:
        resp = client.table("partidos").select("id, nro").in_("nro", nros).execute()
        existentes = {p["nro"]: p["id"] for p in resp.data}

    nuevos = 0
    actualizados = 0
    for r in registros:
        local_id = _get_or_create_equipo(client, torneo_id, r.get("local_nombre"))
        visit_id = _get_or_create_equipo(client, torneo_id, r.get("visitante_nombre"))

        payload = {
            "etapa_id": etapa_id,
            "local_equipo_id": local_id,
            "visitante_equipo_id": visit_id,
            "nro": r.get("nro"),
            "resultado_local": r.get("resultado_local"),
            "resultado_visitante": r.get("resultado_visitante"),
            "puntos_local": r.get("puntos_local"),
            "puntos_visitante": r.get("puntos_visitante"),
            "fecha_hora": str(r["fecha_hora"]) if r.get("fecha_hora") is not None else None,
            "referee": r.get("referee") or "",
            "estado": r.get("estado") or "Pendiente",
        }

        if r.get("nro") and r["nro"] in existentes:
            client.table("partidos").update(payload).eq("id", existentes[r["nro"]]).execute()
            actualizados += 1
        else:
            client.table("partidos").insert(payload).execute()
            nuevos += 1

    return nuevos, actualizados


# ---------- UI ----------

def _tipo_torneo(nombre) -> str:
    """Normaliza el nombre del torneo a su tipo visible (mismo criterio que supabase_client)."""
    nombre = (nombre or "").strip()
    if nombre.startswith("Torneo "):
        return "Clasificatorio"
    return nombre or "Clasificatorio"


def _etiqueta_torneo(row) -> str:
    return f"{row['division']} · {_tipo_torneo(row['nombre'])} ({row['temporada']})"


def _tab_pegar(client):
    st.subheader("📥 Pegar partidos de bd.uar")

    torneos = _get_torneos(client)
    if torneos.empty:
        st.warning("Todavía no hay torneos. Cargalos en la pestaña 'Torneos'.")
        return

    torneos["etiqueta"] = torneos.apply(_etiqueta_torneo, axis=1)
    etiqueta = st.selectbox("Torneo", torneos["etiqueta"].tolist(),
                            index=0, key="pegar_torneo")
    torneo_id = int(torneos.loc[torneos["etiqueta"] == etiqueta, "id"].iloc[0])

    st.info(f"Los partidos se van a guardar en **{etiqueta}**. Verificá que sea el torneo correcto.")

    texto = st.text_area(
        "Pegá acá las filas copiadas de bd.uar (una por línea, separadas por tab):",
        height=200,
        placeholder="314354\tLOS 50\t7\t64 [5]\tMAR DEL PLATA Marron\t18/07/2026 13:30\tDiaz, Tomas\t Cerrado\t\n...",
    )

    st.markdown(
        "Formato: `Nro. \\t Local \\t ResultadoL \\t ResultadoV \\t Visitante \\t Fecha y Hora \\t Referee \\t Estado`"
    )

    if not texto.strip():
        return

    df_parsed = parsear_pegado(texto)
    registros, errores = _puede_guardar_tabla(df_parsed)

    st.markdown(f"Se detectaron **{len(df_parsed)} filas**.")

    if errores:
        with st.expander(f"⚠️ {len(errores)} inconsistencia(s) para revisar", expanded=True):
            for e in errores:
                st.warning(e)

    if df_parsed.empty:
        return

    preview = df_parsed.copy()
    preview["Interpretado"] = preview.apply(
        lambda r: f"→ {r['Local']} {r['ResultadoL'] or '-'} vs {r['ResultadoV'] or '-'} "
                  f"({r['Visitante']})", axis=1)
    st.dataframe(preview[["Nro.", "Interpretado", "Fecha y Hora", "Estado"]],
                 use_container_width=True, hide_index=True)

    if st.button(f"✔ Confirmar carga en {etiqueta}", type="primary"):
        nuevos, actualizados = _persistir_partidos(client, torneo_id, registros)
        st.success(f"✅ {nuevos} partido(s) cargados, {actualizados} actualizado(s).")
        st.cache_data.clear()
        st.rerun()


def _tab_torneos(client):
    st.subheader("🏆 Torneos")
    torneos = _get_torneos(client)

    with st.form("nuevo_torneo"):
        division = st.text_input("División (año de nacimiento, ej: 2010)")
        temporada = st.number_input("Temporada", min_value=2000, max_value=2100, value=2026, step=1)
        tipo = st.selectbox("Tipo de torneo", ["Clasificatorio", "Oro", "Plata", "Regular"])
        corte_top = st.number_input("Corte clasificación (cuántos clasifican)", min_value=1, max_value=50, value=7)
        activo = st.checkbox("Activo")
        enviar = st.form_submit_button("Crear torneo")
        if enviar:
            if not division.strip():
                st.error("Falta la división.")
            else:
                nombre = tipo.strip() or "Clasificatorio"
                existe = client.table("torneos") \
                    .select("id").eq("division", division.strip()) \
                    .eq("temporada", int(temporada)) \
                    .eq("nombre", nombre).execute()
                if existe.data:
                    st.error("Ya existe un torneo con esa división + temporada + tipo.")
                else:
                    torneo_id = client.table("torneos").insert({
                        "nombre": nombre, "division": division.strip(),
                        "temporada": int(temporada), "corte_top": int(corte_top),
                        "activa": activo,
                    }).execute().data[0]["id"]
                    _get_or_create_etapa(client, torneo_id, "Regular")
                    st.success(f"Torneo '{division} {nombre}' creado.")
                    st.rerun()

    if torneos.empty:
        st.info("Todavía no hay torneos.")
        return

    st.markdown("---")
    for _, t in torneos.iterrows():
        with st.expander(f"{t['division']} · {_tipo_torneo(t['nombre'])} ({t['temporada']}) — {'✅ activo' if t['activa'] else 'inactivo'}"):
            c1, c2 = st.columns([1, 1])
            nuevo_corte = c1.number_input(
                "Corte", min_value=1, max_value=50, value=int(t["corte_top"]) if t["corte_top"] else 7,
                key=f"corte_{t['id']}")
            if c1.button("Guardar corte", key=f"btn_corte_{t['id']}"):
                client.table("torneos").update({"corte_top": int(nuevo_corte)}).eq("id", t["id"]).execute()
                st.rerun()
            if c2.button("Activar", key=f"act_{t['id']}"):
                client.table("torneos").update({"activa": True}).eq("id", t["id"]).execute()
                client.table("torneos").update({"activa": False}).ne("id", t["id"]).execute()
                st.rerun()
            etapas = _get_etapas(client, t["id"])
            st.caption("Etapas: " + ", ".join(etapas["nombre"].tolist()) if not etapas.empty else "Sin etapas")


def _tab_editar(client):
    st.subheader("✏️ Editar partido")

    torneos = _get_torneos(client)
    if torneos.empty:
        st.info("No hay torneos aún.")
        return
    torneos["etiqueta"] = torneos.apply(_etiqueta_torneo, axis=1)
    etiqueta = st.selectbox("Torneo", torneos["etiqueta"].tolist(), key="edit_torneo")
    torneo_id = int(torneos.loc[torneos["etiqueta"] == etiqueta, "id"].iloc[0])

    etapa_ids = _get_etapas(client, torneo_id)["id"].tolist()
    if not etapa_ids:
        st.info("Ese torneo no tiene etapas.")
        return

    partidos = client.table("partidos") \
        .select("id, nro, resultado_local, resultado_visitante, puntos_local, "
                "puntos_visitante, fecha_hora, referee, estado, "
                "local_equipo:local_equipo_id(nombre), visitante_equipo:visitante_equipo_id(nombre)") \
        .in_("etapa_id", etapa_ids).order("fecha_hora").execute()

    if not partidos.data:
        st.info("No hay partidos para editar.")
        return

    opciones = {}
    for p in partidos.data:
        loc = p["local_equipo"]["nombre"] if p.get("local_equipo") else "?"
        vis = p["visitante_equipo"]["nombre"] if p.get("visitante_equipo") else "?"
        opciones[f"[{p.get('nro')}] {loc} vs {vis}"] = p

    seleccion = st.selectbox("Partido", list(opciones.keys()))
    p = opciones[seleccion]

    with st.form("editar_partido"):
        c1, c2 = st.columns(2)
        res_l = c1.text_input("ResultadoL", value=str(p["resultado_local"]) if p["resultado_local"] is not None else "")
        res_v = c2.text_input("ResultadoV", value=str(p["resultado_visitante"]) if p["resultado_visitante"] is not None else "")
        c3, c4 = st.columns(2)
        pts_l = c3.number_input("Puntos L", min_value=0, max_value=8, value=int(p["puntos_local"] or 0))
        pts_v = c4.number_input("Puntos V", min_value=0, max_value=8, value=int(p["puntos_visitante"] or 0))
        fecha = st.text_input("Fecha y Hora (DD/MM/AAAA HH:MM)",
                              value=str(p["fecha_hora"]) if p["fecha_hora"] else "")
        estado = st.selectbox("Estado", ["Pendiente", "En Curso", "Cerrado"],
                              index=["Pendiente", "En Curso", "Cerrado"].index(p["estado"] or "Pendiente"))
        guardar = st.form_submit_button("Guardar cambios")
        if guardar:
            payload = {
                "resultado_local": _int_or_none(res_l) if res_l.strip() else None,
                "resultado_visitante": _int_or_none(res_v) if res_v.strip() else None,
                "puntos_local": int(pts_l),
                "puntos_visitante": int(pts_v),
                "fecha_hora": str(_fecha_parse(fecha)) if fecha.strip() else None,
                "estado": estado,
            }
            client.table("partidos").update(payload).eq("id", p["id"]).execute()
            st.success("Partido actualizado.")
            st.cache_data.clear()
            st.rerun()


def _tab_migrar(client):
    import google_sheets_client as gsc
    from logic import parse_resultado as _pr

    st.subheader("🔁 Migrar desde la planilla")
    st.markdown(
        "Lee la planilla actual (Google Sheets) y crea/actualiza torneos, "
        "equipos y partidos en Supabase. Es **idempotente**: si el partido ya "
        "existe (mismo Nro. de bd.uar) lo actualiza en vez de duplicarlo."
    )
    temporada = st.number_input("Temporada de los torneos a migrar",
                                min_value=2000, max_value=2100, value=2026, step=1)

    if not st.button("🚀 Ejecutar migración", use_container_width=True, type="primary"):
        return

    try:
        c_sheets = gsc.get_gspread_client()
    except Exception as e:
        st.error(f"No se pudo conectar a Google Sheets: {e}")
        return

    divisiones = gsc.get_available_birth_years(c_sheets)
    if not divisiones:
        st.warning("No se detectaron divisiones en la planilla.")
        return

    status = st.status(f"Migrando {len(divisiones)} divisiones...", expanded=True)
    resumen = []

    for division in divisiones:
        es_tarjetas = division.endswith("T")

        if es_tarjetas:
            anio = division[:-1]
            df = gsc.get_tarjetas_data(c_sheets, anio)
            if df.empty:
                status.write(f"  • {division}: sin datos de tarjetas")
                continue
            torneo_id = _torneo_por_division(client, anio, int(temporada))
            creadas = 0
            for _, t in df.iterrows():
                eq_id = _get_or_create_equipo(client, torneo_id, t.get("Equipo"))
                if eq_id is None:
                    continue
                client.table("tarjetas").insert({
                    "equipo_id": eq_id,
                    "fecha": _ts_fecha(t.get("Fecha")),
                    "incidencia": t.get("Incidencia"),
                    "instancia": t.get("Instancia"),
                    "rival": t.get("Rival"),
                    "momento": t.get("Momento"),
                    "detalle": t.get("Detalle"),
                }).execute()
                creadas += 1
            status.write(f"  • {division}: {creadas} tarjetas migradas")
            resumen.append((division, creadas, 0))
            continue

        df = gsc.get_division_data(c_sheets, division)
        if df.empty:
            status.write(f"  • {division}: sin datos")
            continue

        torneo_id = _torneo_por_division(client, division, int(temporada))
        etapa_id = _get_or_create_etapa(client, torneo_id, "Regular")

        nuevos = 0
        guardados = 0
        for _, row in df.iterrows():
            local_id = _get_or_create_equipo(client, torneo_id, row.get("Local"))
            visit_id = _get_or_create_equipo(client, torneo_id, row.get("Visitante"))

            res_l, pts_l = _pr(row.get("ResultadoL"))
            res_v, pts_v = _pr(row.get("ResultadoV"))
            estado = (row.get("Estado") or "").strip() or "Pendiente"

            nro = _int_or_none(row.get("Nro."))
            payload = {
                "etapa_id": etapa_id,
                "local_equipo_id": local_id,
                "visitante_equipo_id": visit_id,
                "nro": nro,
                "resultado_local": res_l,
                "resultado_visitante": res_v,
                "puntos_local": pts_l,
                "puntos_visitante": pts_v,
                "fecha_hora": _ts_fecha(row.get("Fecha y Hora")),
                "estado": estado,
            }
            if nro:
                client.table("partidos").upsert(payload, on_conflict="nro").execute()
                nuevos += 1
            else:
                client.table("partidos").insert(payload).execute()
                guardados += 1

        status.write(f"  • {division}: {nuevos} partidos actualizados/guardados")
        resumen.append((division, nuevos, guardados))

    status.update(label="Migración finalizada ✔", state="complete", expanded=True)
    for d, n, g in resumen:
        st.success(f"{d}: {n} partidos (o {g} sin nro)")
    st.success("Listo. Ya podés usar la app sobre Supabase.")
    st.cache_data.clear()
    st.rerun()


def _torneo_por_division(client, division, temporada):
    """Busca o crea el torneo de tipo Clasificatorio para división+temporada."""
    resp = client.table("torneos") \
        .select("id, nombre").eq("division", division) \
        .eq("temporada", temporada).execute()
    if resp.data:
        # Preferir el torneo Clasificatorio (los viejos tienen nombre "Torneo ...")
        for t in resp.data:
            if _es_clasificatorio(t.get("nombre")):
                return t["id"]
        return resp.data[0]["id"]
    res = client.table("torneos").insert({
        "nombre": "Clasificatorio",
        "division": division,
        "temporada": temporada,
        "corte_top": 7,
        "activa": True,
    }).execute()
    return res.data[0]["id"]


def _es_clasificatorio(nombre) -> bool:
    nombre = (nombre or "").strip()
    return nombre == "Clasificatorio" or nombre.startswith("Torneo ")


def _ts_fecha(v):
    """Convierte fecha a texto ISO corto para Supabase (o None)."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    dt = pd.to_datetime(v, dayfirst=True)
    if pd.isna(dt):
        return None
    return str(dt)


def render_admin():
    """Punto de entrada desde la sidebar de la app."""
    st.title("⚙️ Administración")

    if not is_admin_signed_in():
        _render_login()
        return

    if st.session_state.get("su_admin_email"):
        st.caption(f"Logueado como {st.session_state['su_admin_email']}")
    if st.button("Salir"):
        sign_out()

    try:
        client = _admin_client()
    except Exception as e:
        st.error(f"No se pudo conectar: {e}")
        return

    tab_pegar, tab_edit, tab_tor, tab_mig = st.tabs(
        ["📥 Pegar partidos", "✏️ Editar partido", "🏆 Torneos", "🔁 Migrar planilla"])
    with tab_pegar:
        _tab_pegar(client)
    with tab_edit:
        _tab_editar(client)
    with tab_tor:
        _tab_torneos(client)
    with tab_mig:
        _tab_migrar(client)