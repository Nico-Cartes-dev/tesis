"""
Genera DATA SINTÉTICA en data/stock_history.csv + data/alerts.log para probar
EN 2 SEGUNDOS el sistema de forecast (slope), alerta PROACTIVA naranja y el
ranking semanal sin tener que correr detect.py una hora real.

Lo que simula (1 hora de historia, 1 muestra por minuto = 60 filas):
  - camara_02 / repisa_yogures    → BAJADA RÁPIDA de 100% → 40% en 60min (1%/min de bajada)
                                   Tiempo hasta acabarse estimado: otros 40 min (<30? No,
                                   pero con más bajada: 1.5%/min = 26 min → cruza umbral de
                                   alerta anticipada).
  - camara_01 / repisa_leches     → BAJADA LENTA 100% → 88% en 60min (0.2%/min)
  - camara_01 / repisa_gaseosas   → ESTABLE + ligeramente repone (no hay forecast de vacío)

Esto NO borra datos reales: hace backup automático de los CSVs existentes.
Para DESHACER: solo borrás los .csv actuales y renombras los .bak.

Uso:
  python src/prediction/seed_demo_forecast.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "data"
STOCK_CSV = DATA_DIR / "stock_history.csv"
ALERTS_LOG = DATA_DIR / "alerts.log"
HIST_CSV = DATA_DIR / "alerts_historial.csv"

COLS_STOCK = [
    "timestamp_local", "timestamp_unix", "camera_id", "roi_id",
    "pct_ocupado", "pct_vacio", "num_detecciones", "latencia_e2e_s",
    "dia_semana", "hora", "hora_decimal", "es_fin_de_semana",
]
COLS_ALERTAS = [
    "timestamp_local", "timestamp_unix_utc", "camera_id", "roi_id",
    "pct_vacio", "pct_umbral", "metadatos_json",
]
COLS_HIST = [
    "timestamp_local", "camera_id", "roi_id", "pct_vacio", "pct_umbral", "area_roi_px",
]


def backup_si_existe(p: Path):
    if p.exists() and p.stat().st_size > 0:
        bak = p.with_suffix(p.suffix + f".bak.{int(time.time())}")
        shutil.copy2(p, bak)
        print(f"💾 Backup de datos existentes: {bak}")


def init_csv(path: Path, cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(cols)


def muestrear_stock(cam: str, roi: str, ocupado_inicial: float,
                    slope_ocup_por_min: float, n_minutos: int,
                    base_time: datetime, area_px_aprox: int = 40000) -> list[dict]:
    rows = []
    for i in range(n_minutos):
        ts_dt = base_time + timedelta(minutes=i)
        ocup = max(0.0, min(1.0, ocupado_inicial + slope_ocup_por_min * i))
        vacio = 1.0 - ocup
        ts_unix = ts_dt.timestamp()
        n_prod = max(0, int(round(ocup * area_px_aprox / 4000)))
        row = {
            "timestamp_local": ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_unix": f"{ts_unix:.3f}",
            "camera_id": cam,
            "roi_id": roi,
            "pct_ocupado": f"{ocup:.6f}",
            "pct_vacio": f"{vacio:.6f}",
            "num_detecciones": n_prod,
            "latencia_e2e_s": "0.15",
            "dia_semana": int(ts_dt.weekday()),
            "hora": int(ts_dt.hour),
            "hora_decimal": f"{ts_dt.hour + ts_dt.minute/60.0:.4f}",
            "es_fin_de_semana": int(ts_dt.weekday() >= 5),
        }
        rows.append(row)
    return rows


def generar_alertas_ejemplo(base_time: datetime):
    """
    Genera algunas alertas en alerts.log (para alimentar el ranking de riesgo).
    """
    filas = []
    # 3 alertas de yogures (más riesgosa)
    for i, mins in enumerate([60 - 5, 60 - 4, 60 - 3]):
        ts = base_time + timedelta(minutes=mins)
        filas.append((
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            f"{ts.timestamp():.3f}",
            "camara_02",
            "repisa_yogures",
            0.52 + i * 0.02,
            0.35,
            '{"tipo":"REACTIVA_umbral_vacio","pct_ocupado":' + f"{0.48 - i*0.02:.2f}" + '}',
        ))
    # 1 alerta de leches
    ts = base_time + timedelta(minutes=58)
    filas.append((
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        f"{ts.timestamp():.3f}",
        "camara_01",
        "repisa_leches",
        0.36,
        0.35,
        '{"tipo":"REACTIVA_umbral_vacio","pct_ocupado":0.64}',
    ))
    return filas


def main():
    if "--no-backup" not in sys.argv:
        backup_si_existe(STOCK_CSV)
        backup_si_existe(ALERTS_LOG)
        backup_si_existe(HIST_CSV)
    init_csv(STOCK_CSV, COLS_STOCK)
    init_csv(ALERTS_LOG, COLS_ALERTAS)
    init_csv(HIST_CSV, COLS_HIST)

    base_time = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=60)

    escenarios = [
        # (cam_id,    roi_id,         oc0,  slope/min,  minutos,  area)
        ("camara_02", "repisa_yogures", 1.00, -0.015,   60,       50000),
        ("camara_01", "repisa_leches",  1.00, -0.002,   60,       40000),
        ("camara_01", "repisa_gaseosas", 0.70, +0.0005, 60,       45000),
    ]

    total = 0
    with open(STOCK_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for cam, roi, oc0, slope, m, area in escenarios:
            rows = muestrear_stock(cam, roi, oc0, slope, m, base_time, area)
            for r in rows:
                w.writerow([r[c] for c in COLS_STOCK])
            total += len(rows)
            print(f"📦 {cam}/{roi}: {m} muestras ({oc0*100:.0f}% → "
                  f"{max(0, oc0 + slope*m)*100:.0f}%, "
                  f"slope={slope*100:+.2f}%/min)")

    print(f"\n✅ {total} filas de stock_history escritas en {STOCK_CSV}")

    alertas = generar_alertas_ejemplo(base_time)
    with open(ALERTS_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for a in alertas:
            w.writerow(list(a))
    print(f"✅ {len(alertas)} alertas de ejemplo escritas en {ALERTS_LOG}")

    with open(HIST_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for a in alertas:
            w.writerow([a[0], a[2], a[3], f"{a[4]:.4f}", f"{a[5]:.4f}", 45000])
    print(f"✅ Historial LLm actualizado en {HIST_CSV}")

    print("\n" + "=" * 70)
    print("AHORA podés correr esto para ver el sistema en acción:")
    print("=" * 70)
    print("""
# 1) Ver el FORECAST en una imagen cualquiera (sin esperar nada)
#    Toma el histórico que acabamos de escribir y pronostica EN VIVO
#    por cada ROI (solo necesita haber ROIs con el mismo nombre en camaras.yaml)
python src/detection/detect.py --weights models/runs/stock_detector_v1/weights/best.pt \
       --folder data/processed/images/val --camera camara_02
# Resultado esperado en el panel para repisa_yogures:
#   🟠 [PRONÓSTICO: se acaba] ↳ se vacía aproximadamente en 26 min  [método: RollingSlope  ETA HH:MM (t=26min)  conf=0.99]

# 2) Ver el RANKING semanal TOP 5
python src/prediction/top_riesgo.py --exportar-md docs/ranking_riesgo_semanal.md
# Resultado esperado:
#   1. camara_02 / repisa_yogures       slope=-1.500 %/min  ocup_med=55.0%  alertas_7d=3  score=ALTO
#   2. camara_01 / repisa_leches        slope=-0.200 %/min  ocup_med=94.1%  alertas_7d=1  score=MEDIO
#   3. camara_01 / repisa_gaseosas      slope=+0.050 %/min  ocup_med=71.5%  alertas_7d=0  score=BAJO
""")


if __name__ == "__main__":
    main()
