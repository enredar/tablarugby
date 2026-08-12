"""Migración: Google Sheets (legacy) → Supabase.

Puede ejecutarse:
- Como script local: `python migrate_sheets_to_supabase.py`
- Parado en un directorio con .streamlit/secrets.toml (usa st.secrets)

Requiere (en secrets.toml o variables de entorno):
  gcp_service_account  (para leer la planilla)
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY  (para escribir, evita lidiar con RLS aquí)

Genera por cada pestaña de división (ej: 2010) un torneo + etapa
"Regular" + equipos + partidos; y para las pestañas "T" las tarjetas.
"""

import os
import sys

import pandas as pd

# ---------- Configuración desde entorno / secrets ----------

def _secrets_get(key, default=None):
    # Si streamlit está disponible y hay secrets cargados, los usa
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return os.environ.get(key, default)


SUPABASE_URL = _secrets_get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = (_secrets_get("SUPABASE_SERVICE_ROLE_KEY")
                or os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
TEMPORADA = int(os.environ.get("TEMPORADA_MIGRACION", "2026"))


# ---------- Carga desde Google Sheets ----------
try:
    import google_sheets_client as gsc
    from supabase import create_client
    from logic import parse_resultado
except ImportError as e:
    print(f"Falta una dependencia: {e}")
    sys.exit(1)


def cargar_datos_desde_sheets():
    cliente_sheets = gsc.get_gspread_client()
    divisiones = gsc.get_available_birth_years(cliente_sheets)
    # Sólo pestañas "puras" (año) y sus "T"
    divisiones = [d for d in divisiones if d]
    return cliente_sheets, divisiones


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("FALTAN SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    cliente_sheets, divisiones = cargar_datos_desde_sheets()
    print(f"Divisiones detectadas: {divisiones}")

    for division in divisiones:
        es_tarjetas = division.endswith("T")

        if es_tarjetas:
            anio = division[:-1]
            df = gsc.get_tarjetas_data(cliente_sheets, anio)
            if df.empty:
                print(f"  [tarjetas {division}] sin datos, se omite")
                continue
            torneo_id = _torneo_id(supabase, anio)

            for _, t in df.iterrows():
                equipo_id = _equipo_id(supabase, torneo_id, t.get("Equipo"))
                payload = {
                    "equipo_id": equipo_id,
                    "fecha": _fecha(t.get("Fecha")),
                    "incidencia": t.get("Incidencia"),
                    "instancia": t.get("Instancia"),
                    "rival": t.get("Rival"),
                    "momento": t.get("Momento"),
                    "detalle": t.get("Detalle"),
                }
                supabase.table("tarjetas").insert(payload).execute()
            print(f"  [tarjetas {division}] insertadas tarjetas")
            continue

        df = gsc.get_division_data(cliente_sheets, division)
        if df.empty:
            print(f"  [{division}] sin datos, se omite")
            continue

        torneo_id = _torneo_id(supabase, division)
        etapa_id = _etapa_id(supabase, torneo_id, "Regular")

        for _, row in df.iterrows():
            local_id = _equipo_id(supabase, torneo_id, row.get("Local"))
            visit_id = _equipo_id(supabase, torneo_id, row.get("Visitante"))

            res_l, pts_l = parse_resultado(row.get("ResultadoL"))
            res_v, pts_v = parse_resultado(row.get("ResultadoV"))

            # Partido jugado o pendiente
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
                "fecha_hora": _fecha(row.get("Fecha y Hora")),
                "estado": estado,
            }
            # Upsert por nro para que sea idempotente
            if nro:
                supabase.table("partidos").upsert(payload, on_conflict="nro").execute()
            else:
                supabase.table("partidos").insert(payload).execute()

        print(f"  [{division}] torneo + etapa + partidos OK")

    print("Migración terminada.")


def _torneo_id(supabase, division):
    resp = supabase.table("torneos").select("id").eq("division", division) \
        .eq("temporada", TEMPORADA).execute()
    if resp.data:
        return resp.data[0]["id"]
    res = supabase.table("torneos").insert({
        "nombre": f"Torneo {division} {TEMPORADA}",
        "division": division,
        "temporada": TEMPORADA,
        "corte_top": 7,
        "activa": True,
    }).execute()
    return res.data[0]["id"]


def _etapa_id(supabase, torneo_id, nombre):
    resp = supabase.table("etapas").select("id").eq("torneo_id", torneo_id) \
        .eq("nombre", nombre).execute()
    if resp.data:
        return resp.data[0]["id"]
    res = supabase.table("etapas").insert({
        "torneo_id": torneo_id, "nombre": nombre, "orden": 0,
    }).execute()
    return res.data[0]["id"]


def _equipo_id(supabase, torneo_id, nombre):
    nombre = (nombre or "").strip()
    if not nombre:
        return None
    resp = supabase.table("equipos").select("id").eq("torneo_id", torneo_id) \
        .eq("nombre", nombre).execute()
    if resp.data:
        return resp.data[0]["id"]
    res = supabase.table("equipos").insert({
        "torneo_id": torneo_id, "nombre": nombre,
    }).execute()
    return res.data[0]["id"]


def _int_or_none(v):
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return None


def _fecha(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    dt = pd.to_datetime(v, dayfirst=True)
    if pd.isna(dt):
        return None
    return str(dt)


if __name__ == "__main__":
    main()