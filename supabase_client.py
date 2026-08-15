"""Adaptador de datos hacia Supabase (PostgreSQL).

Expone la MISMA interfaz que google_sheets_client.py para que app.py
no necesite reescrituras grandes. Todas las funciones devuelven
DataFrames con exactamente las mismas columnas que hoy.
"""

import streamlit as st
import pandas as pd
from supabase import create_client, Client


@st.cache_resource(show_spinner="Conectando a Supabase...")
def get_source_client() -> Client:
    """Cliente público (anon key + RLS). Para leer lo público."""
    url = st.secrets["SUPABASE_URL"]
    anon_key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, anon_key)


# ---------- Helpers de torneos ----------

def _tipo_torneo(nombre) -> str:
    """Normaliza el nombre del torneo a su tipo visible.

    Los torneos viejos creados por la migración tenían nombre
    "Torneo 2010 2026" → se muestran como "Clasificatorio".
    """
    nombre = (nombre or "").strip()
    if nombre.startswith("Torneo "):
        return "Clasificatorio"
    return nombre or "Clasificatorio"


def _label_torneo(division: str, nombre, temporada=None) -> str:
    tipo = _tipo_torneo(nombre)
    if temporada is None:
        return f"{division} · {tipo}"
    return f"{division} · {tipo} ({temporada})"


def _division_from_label(label: str) -> str | None:
    """Extrae la división (año) de una etiqueta '2010 · Oro (2026)'."""
    parts = label.split("·")
    if not parts:
        return None
    return parts[0].strip()


def _temporada_from_label(label: str) -> int | None:
    """Extrae la temporada de '2010 · Oro (2026)' → 2026."""
    import re
    m = re.search(r"\((\d{4})\)", label)
    return int(m.group(1)) if m else None


# ---------- Torneos ----------

@st.cache_data(ttl="10m", show_spinner="Cargando torneos...")
def get_available_years(_client: Client, incluir_archivados: bool = False) -> list[str]:
    """Lista los torneos disponibles, formateados como '2010 · Oro'.

    Por defecto solo devuelve los torneos activos. Con `incluir_archivados=True`
    devuelve todos (activos + archivados).
    """
    resp = _client.table("torneos").select("*").order("temporada", desc=True).execute()
    df = pd.DataFrame(resp.data)

    if df.empty:
        return []

    if not incluir_archivados:
        df = df[df["activa"] == True]

    # Ordenar: activo primero, luego temporada desc, luego division asc, tipo asc
    df = df.sort_values(
        by=["activa", "temporada", "division", "nombre"],
        ascending=[False, False, True, True],
    )

    return [_label_torneo(row.division, row.nombre, row.temporada) for _, row in df.iterrows()]


def get_default_year(_client: Client) -> str:
    """Devuelve el torneo activo, o el de temporada más reciente."""
    resp = _client.table("torneos").select("*").order("temporada", desc=True).execute()
    df = pd.DataFrame(resp.data)

    if df.empty:
        return None

    if "activa" in df.columns and df["activa"].any():
        df = df.sort_values(["activa", "temporada"], ascending=[False, False])
    else:
        df = df.sort_values("temporada", ascending=False)

    row = df.iloc[0]
    return _label_torneo(row.division, row.nombre, row.temporada)


def get_corte_top(_client: Client, division_label: str) -> int:
    """Devuelve cuántos equipos clasifican para el torneo (para colorear)."""
    torneo_id = _torneo_id_from_label(_client, division_label)
    if torneo_id is None:
        return 7
    resp = _client.table("torneos").select("corte_top").eq("id", torneo_id).execute()
    if not resp.data:
        return 7
    return resp.data[0].get("corte_top") or 7


def _tipo_from_label(label: str) -> str:
    """Extrae el tipo visible de '2010 · Oro (2026)' → 'Oro'."""
    parts = label.split("·")
    if len(parts) != 2:
        return ""
    parte = parts[1].strip()
    import re
    parte = re.sub(r"\(\d{4}\)\s*$", "", parte).strip()
    return _tipo_torneo(parte)


def _torneo_id_from_label(_client: Client, label: str) -> int | None:
    """Parsea '2010 · Oro (2026)' y devuelve el id del torneo correspondiente."""
    parts = label.split("·")
    if len(parts) != 2:
        return None
    division = parts[0].strip()
    tipo = _tipo_from_label(label)
    temporada = _temporada_from_label(label)

    resp = _client.table("torneos") \
        .select("id, nombre, temporada") \
        .eq("division", division) \
        .order("temporada", desc=True) \
        .execute()

    for t in resp.data or []:
        if _tipo_torneo(t.get("nombre")) == tipo:
            if temporada is not None and int(t.get("temporada")) != temporada:
                continue
            return t["id"]
    return None


# ---------- Datos por división ----------

@st.cache_data(ttl="10m", show_spinner="Cargando historial de la división...")
def get_division_history(_client: Client, division_label: str) -> pd.DataFrame:
    """
    Une TODOS los partidos de todos los torneos de la misma división
    (ej: 2010 Clasificatorio + 2010 Oro + 2010 Plata de todas las temporadas).
    Se usa para historial directo y predicciones.

    Devuelve las mismas columnas que get_division_data:
    Nro., Local, ResultadoL, ResultadoV, Visitante, Fecha y Hora, Estado.
    """
    division = _division_from_label(division_label)
    if division is None:
        return pd.DataFrame()

    # Todos los torneos de la división (cualquier temporada/tipo)
    torneos = _client.table("torneos") \
        .select("id").eq("division", division).execute()
    if not torneos.data:
        return pd.DataFrame()
    torneo_ids = [t["id"] for t in torneos.data]

    # Todas las etapas de esos torneos
    etapas = _client.table("etapas") \
        .select("id").in_("torneo_id", torneo_ids).execute()
    etapa_ids = [e["id"] for e in etapas.data]
    if not etapa_ids:
        return pd.DataFrame()

    partidos = _client.table("partidos") \
        .select("id, nro, resultado_local, resultado_visitante, puntos_local, "
                "puntos_visitante, fecha_hora, referee, estado, "
                "local_equipo:local_equipo_id(nombre), visitante_equipo:visitante_equipo_id(nombre)") \
        .in_("etapa_id", etapa_ids) \
        .order("fecha_hora") \
        .execute()

    if not partidos.data:
        return pd.DataFrame()

    rows = []
    for p in partidos.data:
        rows.append({
            "Nro.": p.get("nro"),
            "Local": p["local_equipo"]["nombre"] if p.get("local_equipo") else "",
            "ResultadoL": _formatear_resultado(p.get("resultado_local"), p.get("puntos_local")),
            "ResultadoV": _formatear_resultado(p.get("resultado_visitante"), p.get("puntos_visitante")),
            "Visitante": p["visitante_equipo"]["nombre"] if p.get("visitante_equipo") else "",
            "Fecha y Hora": _formatear_fecha(p.get("fecha_hora")),
            "Estado": p.get("estado") or "",
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl="10m", show_spinner="Cargando datos de la división...")
def get_division_data(_client: Client, division_label: str) -> pd.DataFrame:
    """
    Devuelve el DataFrame con las mismas columnas que antes:
    Nro., Local, ResultadoL, ResultadoV, Visitante, Fecha y Hora, Estado.
    """
    torneo_id = _torneo_id_from_label(_client, division_label)
    if torneo_id is None:
        return pd.DataFrame()

    # Obtener etapas del torneo
    etapas = _client.table("etapas") \
        .select("id") \
        .eq("torneo_id", torneo_id) \
        .execute()
    etapa_ids = [e["id"] for e in etapas.data]
    if not etapa_ids:
        return pd.DataFrame()

    partidos = _client.table("partidos") \
        .select("id, nro, resultado_local, resultado_visitante, puntos_local, "
                "puntos_visitante, fecha_hora, referee, estado, "
                "local_equipo:local_equipo_id(nombre), visitante_equipo:visitante_equipo_id(nombre)") \
        .in_("etapa_id", etapa_ids) \
        .order("fecha_hora") \
        .execute()

    if not partidos.data:
        return pd.DataFrame()

    rows = []
    for p in partidos.data:
        rows.append({
            "Nro.": p.get("nro"),
            "Local": p["local_equipo"]["nombre"] if p.get("local_equipo") else "",
            "ResultadoL": _formatear_resultado(p.get("resultado_local"), p.get("puntos_local")),
            "ResultadoV": _formatear_resultado(p.get("resultado_visitante"), p.get("puntos_visitante")),
            "Visitante": p["visitante_equipo"]["nombre"] if p.get("visitante_equipo") else "",
            "Fecha y Hora": _formatear_fecha(p.get("fecha_hora")),
            "Estado": p.get("estado") or "",
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl="10m", show_spinner="Cargando tarjetas...")
def get_tarjetas_data(_client: Client, division_label: str) -> pd.DataFrame:
    """Devuelve las tarjetas de la división con las mismas columnas que antes.

    Las tarjetas son por DIVISIÓN + AÑO CALENDARIO (no por torneo): se buscan
    por el par (division, temporada) extraído del label '2010 · Oro (2026)'.
    Datos viejos con equipo_id (sin division/temporada) se resuelven buscando
    los equipos de todos los torneos de la división en esa temporada.
    """
    division = _division_from_label(division_label)
    temporada = _temporada_from_label(division_label)
    if division is None:
        return pd.DataFrame()

    # Tarjetas nuevas: identificadas por division + temporada + equipo_nombre
    q = _client.table("tarjetas").select(
        "id, equipo_id, equipo_nombre, division, temporada, fecha, incidencia, "
        "instancia, rival, momento, detalle, jugador, documento"
    ).eq("division", division)
    if temporada is not None:
        q = q.eq("temporada", temporada)
    nuevas = q.execute().data or []

    # Tarjetas viejas: con equipo_id sin division/temporada -> resolver por torneos de la división
    viejas = []
    if temporada is not None:
        torneos = _client.table("torneos") \
            .select("id").eq("division", division).eq("temporada", temporada).execute()
        if torneos.data:
            torneo_ids = [t["id"] for t in torneos.data]
            equipos = _client.table("equipos") \
                .select("id, nombre") \
                .in_("torneo_id", torneo_ids) \
                .execute()
            if equipos.data:
                equipo_by_id = {e["id"]: e["nombre"] for e in equipos.data}
                viejas = _client.table("tarjetas") \
                    .select("id, equipo_id, equipo_nombre, division, temporada, fecha, incidencia, "
                            "instancia, rival, momento, detalle, jugador, documento") \
                    .in_("equipo_id", list(equipo_by_id.keys())) \
                    .is_("division", "null") \
                    .execute().data or []
                for t in viejas:
                    t["equipo_nombre"] = equipo_by_id.get(t.get("equipo_id"), "")

    rows = []
    for t in nuevas + viejas:
        rows.append({
            "Equipo": t.get("equipo_nombre") or "",
            "Fecha": _formatear_fecha(t.get("fecha")),
            "Incidencia": t.get("incidencia") or "",
            "Instancia": t.get("instancia") or "",
            "Rival": t.get("rival") or "",
            "Momento": t.get("momento") or "",
            "Detalle": t.get("detalle") or "",
            "Jugador": t.get("jugador") or "",
            "Documento": t.get("documento") or "",
        })

    return pd.DataFrame(rows, columns=[
        "Equipo", "Fecha", "Incidencia", "Instancia", "Rival", "Momento", "Detalle",
        "Jugador", "Documento",
    ])


# ---------- Helpers ----------

def _formatear_resultado(puntos_tantos, puntos_torneo):
    """Reconstruye el formato '25 [4]' que espera parse_resultado()."""
    if puntos_tantos is None:
        return ""
    if puntos_torneo:
        return f"{puntos_tantos} [{puntos_torneo}]"
    return str(puntos_tantos)


def _formatear_fecha(iso):
    """Convierte una fecha ISO de Supabase a 'DD/MM/YYYY HH:MM' (sin tz)."""
    if not iso:
        return ""
    dt = pd.to_datetime(iso)
    if dt.tzinfo is not None:
        dt = dt.tz_localize(None)
    return dt.strftime("%d/%m/%Y %H:%M")


# Aliases para compatibilidad con google_sheets_client
get_gspread_client = get_source_client
get_available_birth_years = get_available_years