"""
Calcula y exporta el ranking SEMANAL de repisas (ROIs) con mayor riesgo de
ruptura de stock (las que "tienden a acabarse más rápido").

Uso:
    # ranking por consola
    python src/prediction/top_riesgo.py

    # ranking + markdown en docs/ranking_riesgo_semanal.md
    python src/prediction/top_riesgo.py --exportar-md docs/ranking_riesgo_semanal.md

Los datos los lee de data/stock_history.csv + data/alerts.log (ForecastService
expone ya el ranking; este script sólo formatea y exporta).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

from prediction.forecast_service import ForecastService  # noqa: E402


def tabla_markdown(filas: list[dict]) -> str:
    if not filas:
        return "_No hay datos suficientes esta semana para generar ranking._"
    cols = ["#", "camara", "roi", "muestras_hist", "slope_medio (%/min)",
            "ocup_media (%)", "alertas_7d", "score_riesgo"]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for i, r in enumerate(filas, start=1):
        lines.append(
            f"| {i} "
            f"| {r['camera_id']} | {r['roi_id']} "
            f"| {r['muestras_hist']} "
            f"| {r['slope_medio_%_por_min']:+.3f} "
            f"| {r['ocupacion_media_%']:.1f} "
            f"| {r['alertas_7d']} "
            f"| **{r['score_riesgo']:.1f}** |"
        )
    return "\n".join(lines) + "\n"


def encabezado_md(top_n: int) -> str:
    ahora = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# Ranking Semanal — Repisas con mayor riesgo de ruptura de stock\n\n"
        f"Generado: {ahora}\n\n"
        f"Metodología: `score_riesgo = -slope*100 + (1 - ocup_media)*30 + alertas_7d*20`\n\n"
        f"TOP {top_n}:\n\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--exportar-md", type=str, default=None,
                    help="Path de salida (.md) del ranking. Ej: docs/ranking_riesgo_semanal.md")
    args = ap.parse_args()

    fs = ForecastService()
    filas = fs.ranking_riesgo_ultima_semana(top_n=args.top)

    print()
    print(f"TOP {args.top} repisas en riesgo de ruptura (últimos 7 días):")
    print("-" * 90)
    for i, r in enumerate(filas, start=1):
        print(
            f"{i:>2}. {r['camera_id']:>12s} / {r['roi_id']:<20s}"
            f"  slope={r['slope_medio_%_por_min']:+.3f} %/min"
            f"  ocup_med={r['ocupacion_media_%']:5.1f}%"
            f"  alertas_7d={r['alertas_7d']:>2d}"
            f"  score={r['score_riesgo']:.1f}"
        )
    if not filas:
        print("(sin datos suficientes todavía: tenés que correr detect.py en modo video")
        print(" por un par de horas para que se acumule stock_history.csv)")

    if args.exportar_md:
        out_p = BASE / args.exportar_md
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(encabezado_md(args.top))
            f.write(tabla_markdown(filas))
            f.write("\n> Actualizar con: `python src/prediction/top_riesgo.py "
                    "--exportar-md docs/ranking_riesgo_semanal.md`\n")
        print(f"\n📄 Ranking exportado a: {out_p}")


if __name__ == "__main__":
    main()
