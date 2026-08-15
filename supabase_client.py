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


def _torneo_id_from_label(_client: Client, label: str) -> int | None:
    """Parsea '2010 · Oro (2026)' y devuelve el id del torneo correspondiente."""
    parts = label.split("·")
    if len(parts) != 2:
        return None
    division = parts[0].strip()
    tipo = _tipo_torneo(parts[1])
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
    """Devuelve las tarjetas de la división con las mismas columnas que antes."""
    torneo_id = _torneo_id_from_label(_client, division_label)
    if torneo_id is None:
        return pd.DataFrame()

    # Equipos del torneo
    equipos = _client.table("equipos") \
        .select("id, nombre") \
        .eq("torneo_id", torneo_id) \
        .execute()
    equipo_by_id = {e["id"]: e["nombre"] for e in equipos.data}
    if not equipo_by_id:
        return pd.DataFrame()

    tarjetas = _client.table("tarjetas") \
        .select("id, equipo_id, fecha, incidencia, instancia, rival, momento, detalle") \
        .in_("equipo_id", list(equipo_by_id.keys())) \
        .execute()

    rows = []
    for t in tarjetas.data:
        rows.append({
            "Equipo": equipo_by_id.get(t.get("equipo_id"), ""),
            "Fecha": _formatear_fecha(t.get("fecha")),
            "Incidencia": t.get("incidencia") or "",
            "Instancia": t.get("instancia") or "",
            "Rival": t.get("rival") or "",
            "Momento": t.get("momento") or "",
            "Detalle": t.get("detalle") or "",
        })

    return pd.DataFrame(rows, columns=[
        "Equipo", "Fecha", "Incidencia", "Instancia", "Rival", "Momento", "Detalle",
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