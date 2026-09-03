"""
Logger persistente del estado de stock por ROI a lo largo del tiempo.

Cada N minutos (configurable, default 60s = 1 min) guarda una fila en:
    data/stock_history.csv
con el nivel de stock medido. Este CSV alimenta a los modelos de
forecasting (RollingSlope, Prophet/SARIMA).
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BASE = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = BASE / "data" / "stock_history.csv"

CSV_COLUMNAS = [
    "timestamp_local",
    "timestamp_unix",
    "camera_id",
    "roi_id",
    "pct_ocupado",
    "pct_vacio",
    "num_detecciones",
    "latencia_e2e_s",
    "dia_semana",          # 0=lunes .. 6=domingo (feature para Prophet/SARIMA)
    "hora",                # 0..23
    "hora_decimal",        # 17.5 = 17:30 (feature slope/estacional)
    "es_fin_de_semana",
]


@dataclass
class MuestraStock:
    camera_id: str
    roi_id: str
    pct_ocupado: float
    pct_vacio: float
    num_detecciones: int = 0
    latencia_e2e_s: float = 0.0
    timestamp_unix: float | None = None  # None = "ahora"

    def as_fila_csv(self) -> list:
        ts = self.timestamp_unix or time.time()
        local = dt.datetime.fromtimestamp(ts)
        return [
            local.strftime("%Y-%m-%d %H:%M:%S"),
            f"{ts:.3f}",
            self.camera_id,
            self.roi_id,
            f"{self.pct_ocupado:.6f}",
            f"{self.pct_vacio:.6f}",
            int(self.num_detecciones),
            f"{self.latencia_e2e_s:.4f}",
            int(local.weekday()),
            int(local.hour),
            round(local.hour + local.minute / 60.0 + local.second / 3600.0, 4),
            int(local.weekday() >= 5),
        ]


class StockHistoryLogger:
    def __init__(
        self,
        csv_path: Path | str = DEFAULT_CSV,
        intervalo_por_roi_seg: float = 60.0,  # 1 min por defecto
    ):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.intervalo = float(intervalo_por_roi_seg)
        self._ultimo_guardado: dict[tuple[str, str], float] = {}
        self._init_csv_si_hace_falta()

    def _init_csv_si_hace_falta(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNAS)

    def deberia_guardar(self, cam: str, roi: str, ahora: float | None = None) -> bool:
        ahora = time.time() if ahora is None else float(ahora)
        clave = (cam, roi)
        ultimo = self._ultimo_guardado.get(clave, 0.0)
        return (ahora - ultimo) >= self.intervalo

    def loggear(self, muestra: MuestraStock) -> bool:
        """Escribe una fila SI pasó el intervalo. Devuelve True si escribió."""
        ts = float(muestra.timestamp_unix or time.time())
        if not self.deberia_guardar(muestra.camera_id, muestra.roi_id, ahora=ts):
            return False

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(muestra.as_fila_csv())
        try:
            f.flush()
        except Exception:
            pass
        self._ultimo_guardado[(muestra.camera_id, muestra.roi_id)] = ts
        return True

    # -------- helpers para leer el histórico --------

    def leer_roi(
        self, camera_id: str, roi_id: str,
        ultimas_filas: int | None = None,
    ) -> list[dict]:
        if not self.csv_path.exists():
            return []
        out: list[dict] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["camera_id"] == camera_id and r["roi_id"] == roi_id:
                    out.append(r)
        if ultimas_filas is not None:
            out = out[-ultimas_filas:]
        return out

    def listar_camaras_y_rois(self) -> list[tuple[str, str]]:
        if not self.csv_path.exists():
            return []
        pares = set()
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                pares.add((r["camera_id"], r["roi_id"]))
        return sorted(pares)


if __name__ == "__main__":  # Smoke test
    logger = StockHistoryLogger(intervalo_por_roi_seg=0.0)
    ok = logger.loggear(MuestraStock(
        camera_id="demo",
        roi_id="prueba",
        pct_ocupado=0.65,
        pct_vacio=0.35,
        num_detecciones=10,
        latencia_e2e_s=0.18,
    ))
    print(f"Guardó = {ok}. Path: {logger.csv_path}")
