"""
"""

from datetime import datetime, timedelta
from langchain_core.tools import tool



@tool
def get_current_datetime(format: str = "full") -> str:
    """
    """
    now = datetime.now()

    formats = {
        "full":  now.strftime("%A %d de %B de %Y, %H:%M hs"),
        "date":  now.strftime("%Y-%m-%d"),
        "short": now.strftime("%d/%m/%Y"),
    }

    return formats.get(format, formats["full"])


@tool
def calculate_study_dates(weeks: int = 4, start_offset_days: int = 0) -> str:
    """
    """
    today = datetime.now()
    start = today + timedelta(days=start_offset_days)

    blocks = []
    for i in range(weeks):
        week_start = start + timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)
        blocks.append(
            f"Semana {i+1}: {week_start.strftime('%d/%m/%Y')} → {week_end.strftime('%d/%m/%Y')}"
        )

    result = f"Plan de {weeks} semana(s) desde {start.strftime('%d/%m/%Y')}:\n"
    result += "\n".join(blocks)
    return result


@tool 
def days_until(target_date_str: str) -> str:
    """
    """
    try:

        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                target = datetime.strptime(target_date_str, fmt)
                break
            except ValueError:
                continue
        else:
            return f"Formato de fecha no reconocido: {target_date_str}. Usa YYYY-MM-DD o DD/MM/YYYY."

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        delta = target - today
        days = delta.days

        if days < 0:
            return f"La fecha {target_date_str} ya pasó (hace {abs(days)} días)."
        
        weeks = days // 7
        remaining_days = days % 7

        return (
            f"Faltan {days} días hasta {target_date_str}.\n"
            f"Equivale a {weeks} semana(s) y {remaining_days} día(s) adicionales."
        )

    except Exception as e:
        return f"Error al calcular fecha: {str(e)}"