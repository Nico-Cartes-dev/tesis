"""
Sistema de alertas único para el detector de stock.

Responsabilidades:
  - Registrar cada activación (archivo CSV/log).
  - Evitar alertas duplicadas consecutivas sobre el mismo ROI (histeresis).
  - Disparar callbacks por consola y, a futuro, Telegram/email/webhook.

Uso mínimo:
    from alerts.alert_manager import AlertManager
    am = AlertManager(log_path="data/alerts.log")
    am.alertar(camera_id="camara_01", roi_id="repisa_leches",
               pct_vacio=0.45, pct_umbral=0.35, metadatos={"detalles": "..."})
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class AlertaEvento:
    timestamp: float
    camera_id: str
    roi_id: str
    pct_vacio: float
    pct_umbral: float
    metadatos: dict = field(default_factory=dict)

    @property
    def mensaje(self) -> str:
        return (
            f"[ALERTA STOCK] camera={self.camera_id}  roi={self.roi_id}  "
            f"vacío={self.pct_vacio*100:.1f}% (umbral={self.pct_umbral*100:.0f}%)"
        )


class AlertManager:
    """Punto único de salida para todas las alarmas del sistema."""

    def __init__(
        self,
        log_path: str | Path | None = None,
        cooldown_por_roi_seg: float = 60.0,
        log_level: int = logging.INFO,
        extra_callback: Callable[[AlertaEvento], None] | None = None,
    ):
        self.cooldown = cooldown_por_roi_seg
        self.extra_callback = extra_callback
        self._ultima_alerta: dict[str, float] = {}

        # Logger consola
        self.logger = logging.getLogger("shelf_alert")
        if not self.logger.handlers:
            self.logger.setLevel(log_level)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                              datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(ch)

        # Archivo CSV
        self.csv_path: Path | None = None
        self._csv_f = None
        self._csv_w = None
        if log_path is not None:
            self.csv_path = Path(log_path)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            ya_existe = self.csv_path.exists()
            self._csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_w = csv.writer(self._csv_f)
            if not ya_existe or self.csv_path.stat().st_size == 0:
                self._csv_w.writerow([
                    "timestamp_utc", "timestamp_local", "camera_id",
                    "roi_id", "pct_vacio", "pct_umbral", "metadatos",
                ])

    # ------------------------------------------------------------------
    def alertar(
        self,
        camera_id: str,
        roi_id: str,
        pct_vacio: float,
        pct_umbral: float,
        metadatos: dict | None = None,
    ) -> AlertaEvento | None:
        """
        Dispara la alerta si:
          - ya pasó el cooldown para este (camera, roi),
          - y pct_vacio >= pct_umbral.
        Devuelve el evento o None si se suprimió por cooldown.
        """
        if pct_vacio < pct_umbral:
            return None

        clave = f"{camera_id}::{roi_id}"
        ahora = time.time()
        ultima = self._ultima_alerta.get(clave, 0.0)
        if (ahora - ultima) < self.cooldown:
            return None
        self._ultima_alerta[clave] = ahora

        evento = AlertaEvento(
            timestamp=ahora,
            camera_id=camera_id,
            roi_id=roi_id,
            pct_vacio=float(pct_vacio),
            pct_umbral=float(pct_umbral),
            metadatos=dict(metadatos or {}),
        )

        # 1) consola
        self.logger.warning(evento.mensaje)

        # 2) CSV
        if self._csv_w is not None:
            self._csv_w.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(evento.timestamp)),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(evento.timestamp)),
                evento.camera_id,
                evento.roi_id,
                f"{evento.pct_vacio:.6f}",
                f"{evento.pct_umbral:.6f}",
                str(evento.metadatos),
            ])
            try:
                self._csv_f.flush()
            except Exception:
                pass

        # 3) callback del usuario (para Telegram/email/etc.)
        if self.extra_callback is not None:
            try:
                self.extra_callback(evento)
            except Exception as e:  # pragma: no cover - protegemos al pipeline
                self.logger.error("Error en extra_callback: %s", e)

        return evento

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._csv_f is not None:
            try:
                self._csv_f.close()
            finally:
                self._csv_f = None
                self._csv_w = None

    def __enter__(self) -> "AlertManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
