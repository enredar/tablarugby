"""Lógica compartida entre la app y el panel de administración."""

import re


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