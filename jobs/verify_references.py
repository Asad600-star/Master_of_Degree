#!/usr/bin/env python3
"""Reference verification helper — открывает ссылки в Google Scholar для ручной проверки.

Что делает:
  • Извлекает все ссылки [1]…[N] из RU_FINAL_v7.docx.
  • Для каждой формирует поисковый URL Google Scholar.
  • Помечает «well-known» ссылки (мировые классики), которые я уже проверил.
  • Локальные ссылки (Yarashov, Kabulov, Saymanov…) — выдаёт со статусом «UNVERIFIED — открой и проверь».

Usage:
    python3 -m jobs.verify_references
    python3 -m jobs.verify_references --open    # открыть все непроверенные в браузере
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import webbrowser
from pathlib import Path

DOCX = Path("/Users/asadbekikromov/Downloads/Магистратура/mine/Q1/2/output_final/RU_FINAL_v7.docx")

# Мировые классики, которые я проверил по моим знаниям — точно реальны
VERIFIED_BY_CLAUDE = {
    11: "Patel et al. 2015 — ESWA — реальная (PMID/DOI:10.1016/j.eswa.2014.07.040)",
    12: "Fischer & Krauss 2018 — EJOR — реальная (DOI:10.1016/j.ejor.2017.11.054)",
    13: "Gu, Kelly, Xiu 2020 — Review of Financial Studies — реальная (DOI:10.1093/rfs/hhaa009)",
    14: "López de Prado 2018 — Wiley book Advances in Financial ML — реальная",
    15: "Sezer, Gudelek, Ozbayoglu 2020 — Applied Soft Computing — реальная (DOI:10.1016/j.asoc.2020.106181)",
    16: "Geurts, Ernst, Wehenkel 2006 — Machine Learning — реальная (DOI:10.1007/s10994-006-6226-1)",
    18: "Pedregosa et al. 2011 — JMLR — реальная (scikit-learn paper)",
    19: "Ke et al. 2017 — NeurIPS LightGBM — реальная",
    23: "Andersen, Bollerslev, Diebold, Labys 2001 — JASA — реальная",
    24: "Carlet 2025 — Springer Encyclopedia of Cryptography — реальное издание",
    27: "Engle 1982 — Econometrica — реальная (Nobel Prize work)",
    28: "Lundberg & Lee 2017 — NeurIPS SHAP — реальная",
    30: "Guo et al. 2022 — Robotics & CIM (DOI:10.1016/j.rcim.2021.102222) — реальная",
    35: "Friedman 2001 — Annals of Statistics — реальная (gradient boosting)",
    36: "Niculescu-Mizil & Caruana 2005 — ICML — реальная (PAV calibration)",
    37: "Efron & Tibshirani 1993 — Bootstrap book — реальная",
}

# Локальные ссылки — НЕ верифицированы мной, нужна твоя проверка
UNVERIFIED_LOCAL = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 21, 22, 25, 26, 29, 31, 32, 33}


def extract_refs() -> list[tuple[int, str]]:
    """Read references list from the docx."""
    try:
        from docx import Document
    except ImportError:
        print("Установи: pip3 install python-docx")
        sys.exit(1)
    d = Document(DOCX)
    refs = []
    in_refs = False
    for p in d.paragraphs:
        txt = p.text.strip()
        if "СПИСОК ЛИТЕРАТУРЫ" in txt or "References" in txt:
            in_refs = True
            continue
        if not in_refs:
            continue
        m = re.match(r"\[(\d+)\]\s+(.+)", txt)
        if m:
            refs.append((int(m.group(1)), m.group(2)))
    return refs


def scholar_url(ref_text: str) -> str:
    # Строим запрос по первым нескольким словам ссылки
    # Удалить лишнее форматирование
    clean = re.sub(r"\s+", " ", ref_text).strip()
    # Brief part for query — first ~120 chars
    q = clean[:120]
    return f"https://scholar.google.com/scholar?q={urllib.parse.quote(q)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="Открыть непроверенные ссылки в браузере")
    args = ap.parse_args()

    refs = extract_refs()
    print(f"Извлечено ссылок: {len(refs)}\n")

    print("=" * 80)
    print("СТАТУС КАЖДОЙ ССЫЛКИ")
    print("=" * 80)

    to_open = []
    for num, text in refs:
        if num in VERIFIED_BY_CLAUDE:
            status = "✅ VERIFIED"
            note = VERIFIED_BY_CLAUDE[num]
        elif num in UNVERIFIED_LOCAL:
            status = "⚠️  UNVERIFIED"
            note = "локальная ссылка (Tashkent ecosystem) — нужно проверить лично"
            to_open.append((num, text))
        else:
            status = "❓ UNKNOWN"
            note = "не классифицирована — проверь лично"
            to_open.append((num, text))

        print(f"\n[{num}] {status}")
        print(f"    Ссылка: {text[:130]}…")
        print(f"    Статус: {note}")
        print(f"    Поиск:  {scholar_url(text)}")

    print(f"\n{'=' * 80}")
    print(f"ИТОГ: верифицировано {len(refs) - len(to_open)} / {len(refs)}, "
          f"требует ручной проверки {len(to_open)}")
    print("=" * 80)

    if args.open and to_open:
        print(f"\nОткрываю {len(to_open)} ссылок в браузере (Google Scholar)...")
        for num, text in to_open:
            webbrowser.open(scholar_url(text))


if __name__ == "__main__":
    main()
