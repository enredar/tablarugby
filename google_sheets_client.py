# google_sheets_client.py
import streamlit as st
import gspread
from gspread_pandas import Spread, conf
import pandas as pd
from google.oauth2.service_account import Credentials

# ID de tu Hoja de Cálculo de Google. Lo encuentras en la URL:
# https://docs.google.com/spreadsheets/d/ESTE_ES_EL_ID/edit#gid=0
SPREADSHEET_ID = "1W3TjhS7WL4MnvJabFEFlaKtGC8DZKIvlI4EkB3zlCGU" 

@st.cache_resource(show_spinner="Conectando a Google Sheets...")
def get_gspread_client() -> Spread:
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Crear credenciales con scopes personalizados
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )

    return Spread(SPREADSHEET_ID, creds=creds)

@st.cache_data(ttl="10m", show_spinner="Cargando datos de la división...")
def get_division_data(_spreadsheet_client: Spread, anio_nacimiento: str) -> pd.DataFrame:
    """
    Obtiene los datos de una pestaña específica (año de nacimiento) como un DataFrame.
    Retorna un DataFrame vacío si la pestaña no existe o hay un error.
    """
    try:
        sheet_name = str(anio_nacimiento)
        df = _spreadsheet_client.sheet_to_df(sheet=sheet_name, index=None)

        # Si hay una columna de fechas y se necesita tratamiento adicional, hacerlo fuera de esta función
        return df

    except gspread.exceptions.WorksheetNotFound:
        st.error(f"No se encontró la pestaña para el año de nacimiento: {anio_nacimiento}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error al cargar datos de Google Sheets para {anio_nacimiento}: {e}")
        return pd.DataFrame()

def get_available_birth_years(_spreadsheet_client: Spread) -> list[str]:
    """Obtiene la lista de nombres de las pestañas (años de nacimiento)."""
    try:
        return [sheet.title for sheet in _spreadsheet_client.sheets]
    except Exception as e:
        st.warning(f"No se pudieron obtener las divisiones (pestañas) de Google Sheets: {e}")
        return ["2010", "2009", "2008", "2011", "2012"]  # Fallback