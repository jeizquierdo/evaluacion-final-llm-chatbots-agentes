"""tools.datetime

Date/time helper tools used by agents. These functions are exposed as
callable tools for language models and provide simple utilities to obtain
the current date/time, compute a study schedule, and count days until a
target date.

All functions return human-readable strings (the project uses Spanish
messages in the returns). The functions are lightweight and safe to call
from LLM tool loops.
"""

from datetime import datetime, timedelta
from langchain_core.tools import tool



@tool
def get_current_datetime(format: str = "full") -> str:
    """Return the current date/time formatted in one of several styles.

    Args:
        format: one of "full", "date", or "short" selecting the output
            format. Defaults to "full".

    Returns:
        A localized date/time string (Spanish phrasing in this project).
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
    """Generate a simple per-week study schedule.

    Args:
        weeks: number of weeks to include in the plan (default 4).
        start_offset_days: days to offset the start date from today (default 0).

    Returns:
        A multiline string describing the week ranges in Spanish.
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
    """Compute days (and weeks + days) from today until target date.

    The function accepts target dates in either ISO format (YYYY-MM-DD)
    or DD/MM/YYYY. Returned messages are in Spanish to match the project's
    user-facing text.

    Args:
        target_date_str: date string in YYYY-MM-DD or DD/MM/YYYY.

    Returns:
        A human-readable Spanish message describing the time until the date
        or an error message when parsing fails.
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