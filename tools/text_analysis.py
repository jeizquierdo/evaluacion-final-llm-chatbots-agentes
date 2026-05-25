"""tools.text_analysis

Objective text-quality analysis helpers exposed as LLM tools. These functions
provide automated checks (length, section presence, bullet counts) and are
intended to be called by the `validator` agent to produce measurable verdicts.
"""

import re
from langchain_core.tools import tool




EXPECTED_SECTIONS = {
    "plan": [
        "objetivo",
        "semana",         
        "actividades",
        "recursos",
        "fuentes",
    ],
    "explanation": [
        "qué es",
        "cómo funciona",
        "ejemplo",
        "conexión",
        "fuentes",
    ],
    "summary": [
        "idea central",
        "puntos clave",
        "conclusión",
        "fuentes",
    ],
}

MIN_WORDS = {
    "plan":        200,
    "explanation": 150,
    "summary":     100,
}

MAX_WORDS = {
    "plan":        1500,
    "explanation": 1200,
    "summary":     800,
}


@tool
def analyze_output_quality(output_text: str, output_type: str) -> str:
    """Analyze structural and length quality of generated outputs.

    Returns a multi-line Spanish report indicating issues and metrics.
    """
    if output_type not in EXPECTED_SECTIONS:
        return f"Tipo de output no reconocido: '{output_type}'. Usa: plan, explanation, summary."

    text_lower = output_text.lower()
    words = len(output_text.split())
    lines = len([l for l in output_text.splitlines() if l.strip()])

    min_w = MIN_WORDS[output_type]
    max_w = MAX_WORDS[output_type]
    length_ok = min_w <= words <= max_w
    length_note = (
        f"✅ Longitud OK ({words} palabras)"
        if length_ok
        else (
            f"⚠️ Demasiado corto ({words} palabras, mínimo {min_w})"
            if words < min_w
            else f"⚠️ Demasiado largo ({words} palabras, máximo {max_w})"
        )
    )

    expected = EXPECTED_SECTIONS[output_type]
    found = [s for s in expected if s in text_lower]
    missing = [s for s in expected if s not in text_lower]

    sections_ok = len(missing) == 0
    sections_note = (
        f"✅ Todas las secciones presentes: {found}"
        if sections_ok
        else f"⚠️ Secciones faltantes: {missing} | Encontradas: {found}"
    )

    bullet_count = len(re.findall(r"^[\-\*•]\s", output_text, re.MULTILINE))
    bullets_note = f"📋 Bullets/items encontrados: {bullet_count}"


    has_sources = "fuentes" in text_lower or "http" in text_lower or "www." in text_lower
    sources_note = "✅ Sección de fuentes presente" if has_sources else "⚠️ No se detectó sección de fuentes"


    issues = []
    if not length_ok:
        issues.append("longitud")
    if not sections_ok:
        issues.append("secciones faltantes")
    if not has_sources:
        issues.append("sin fuentes")

    verdict = "✅ Output aprobado" if not issues else f"❌ Problemas detectados: {', '.join(issues)}"

    report = f"""
=== Análisis objetivo: {output_type.upper()} ===

{verdict}

--- Métricas ---
{length_note}
Líneas no vacías: {lines}
{bullets_note}

--- Estructura ---
{sections_note}
{sources_note}
""".strip()

    return report


@tool
def count_sections(output_text: str) -> str:
    """Return a short report of markdown-style headings found in the text.

    Useful to detect missing structural sections expected by the validator.
    """
    headings = re.findall(r"^(#{1,3})\s+(.+)$", output_text, re.MULTILINE)

    if not headings:
        return "No se encontraron encabezados markdown (# / ## / ###) en el texto."

    result = f"Secciones encontradas ({len(headings)}):\n"
    for hashes, title in headings:
        level = len(hashes)
        indent = "  " * (level - 1)
        result += f"{indent}H{level}: {title.strip()}\n"

    return result.strip()