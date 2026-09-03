"""
Servicio unificado de forecasting.

Responsabilidades:
  1) Cargar el CSV histórico de stock.
  2) Elegir el forecaster apropiado por cada (cámara, ROI):
     - Si hay >= 200 filas → capa 2 (Prophet / SARIMA / fallback Slope)
     - Sino → capa 1 (RollingSlopeForecaster, sirve con 2+ puntos)
  3) Guardar / cargar modelos entrenados en `models/forecasters/<roi_key>.pkl`
  4) Exponer `forecast_one()` / `forecast_all()` que devuelven ForecastResult listo
     para pintar en el panel de OpenCV o enviar al AlertManager.

Uso en detect.py:
    from prediction.forecast_service import ForecastService
    fc = ForecastService()
    fc.recalcular_para_roi(...)       # cada X minutos
    r = fc.predecir_ahora(roi_key, ocup_actual, ...)
"""

from __future__ import annotations

import datetime as dt
import pickle
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .forecast_models import (
    ForecastResult,
    ProphetForecaster,
    RollingSlopeForecaster,
    ForecasterABC,
)
from .stock_history import DEFAULT_CSV, MuestraStock, StockHistoryLogger

BASE = Path(__file__).resolve().parent.parent.parent
FORECASTERS_DIR = BASE / "models" / "forecasters"


def _roi_key(cam_id: str, roi_id: str) -> str:
    return f"{cam_id}::{roi_id}"


class ForecastService:
    def __init__(
        self,
        csv_path: Path | str = DEFAULT_CSV,
        modelo_cap2_cls: type = ProphetForecaster,
        min_filas_cap2: int = 200,
        horizonte_default_min: int = 180,   # predice 3h al futuro
        recalentar_cada_seg: float = 10 * 60,  # retrain CAP2 cada 10 min
    ):
        self.csv_path = Path(csv_path)
        self.logger = StockHistoryLogger(csv_path=csv_path)
        self.cap2_cls = modelo_cap2_cls
        self.min_filas_cap2 = int(min_filas_cap2)
        self.horizonte_default = int(horizonte_default_min)
        self.recalentar_cada = float(recalentar_cada_seg)

        FORECASTERS_DIR.mkdir(parents=True, exist_ok=True)
        # estado en memoria
        self._forecasters: dict[str, ForecasterABC] = {}
        self._ultimo_calculo_por_roi: dict[str, float] = {}
        self._ultimo_resultado_por_roi: dict[str, ForecastResult] = {}

    # ------------------------------------------------------------------ 1) log

    def loggear_muestra(self, muestra: MuestraStock) -> bool:
        """Delega en StockHistoryLogger (cada 1min por defecto)."""
        return self.logger.loggear(muestra)

    # ------------------------------------------------------------------ 2) data prep

    def _cargar_serie(self, cam_id: str, roi_id: str,
                      ultimas_filas: int | None = None):
        filas = self.logger.leer_roi(cam_id, roi_id, ultimas_filas=ultimas_filas)
        ts: list[float] = []
        oc: list[float] = []
        for r in filas:
            try:
                ts.append(float(r["timestamp_unix"]))
                oc.append(float(r["pct_ocupado"]))
            except Exception:
                continue
        return ts, oc

    # ------------------------------------------------------------------ 3) entrenar / guardar

    def _ruta_pkl(self, roi_key: str) -> Path:
        nombre_safe = roi_key.replace("::", "__").replace("/", "_").replace("\\", "_")
        return FORECASTERS_DIR / f"{nombre_safe}.pkl"

    def _guardar_pkl(self, roi_key: str, forecaster: ForecasterABC) -> None:
        try:
            with open(self._ruta_pkl(roi_key), "wb") as f:
                pickle.dump(forecaster, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _cargar_pkl_si_existe(self, roi_key: str) -> ForecasterABC | None:
        p = self._ruta_pkl(roi_key)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def recalcular_para_roi(
        self,
        cam_id: str,
        roi_id: str,
        forzar: bool = False,
    ) -> ForecasterABC | None:
        """
        Ajusta / reajusta el modelo de una ROI. No hace falta llamarlo en cada
        frame: cada 10min alcanza (controlado por `recalentar_cada_seg`).
        """
        key = _roi_key(cam_id, roi_id)
        ahora = time.time()
        ultimo = self._ultimo_calculo_por_roi.get(key, 0.0)
        if not forzar and (ahora - ultimo) < self.recalentar_cada:
            return self._forecasters.get(key)

        ts, ocup = self._cargar_serie(cam_id, roi_id)
        if len(ts) < 2:
            return None

        # Elegimos modelo según cantidad de datos
        if len(ts) >= self.min_filas_cap2:
            fc: ForecasterABC = self._cargar_pkl_si_existe(key)
            # Si es Cap2 y tiene suficientes datos, re-entrenamos. Para evitar
            # costo excesivo reusamos si corresponde y actualizamos la versión.
            if not isinstance(fc, self.cap2_cls):
                fc = self.cap2_cls(min_filas_para_prophet=self.min_filas_cap2)
            fc.ajustar(ts, ocup)
            self._guardar_pkl(key, fc)
        else:
            fc = RollingSlopeForecaster(ventana_puntos=min(30, len(ts)))
            fc.ajustar(ts, ocup)

        self._forecasters[key] = fc
        self._ultimo_calculo_por_roi[key] = ahora
        return fc

    # ------------------------------------------------------------------ 4) predecir

    def predecir_ahora(
        self,
        cam_id: str,
        roi_id: str,
        pct_ocupado_actual: float,
        horizonte_min: int | None = None,
        recalcular_si_hace_falta: bool = True,
    ) -> ForecastResult:
        key = _roi_key(cam_id, roi_id)
        if recalcular_si_hace_falta:
            self.recalcular_para_roi(cam_id, roi_id)

        fc = self._forecasters.get(key)
        if fc is None:
            vacio = 1.0 - float(pct_ocupado_actual)
            return ForecastResult(
                roi_key=key,
                pct_ocupado_actual=float(pct_ocupado_actual),
                pct_vacio_actual=vacio,
                tiempo_hasta_vacio_min=None,
                fecha_estimada_vacio=None,
                horizonte_minutos=int(horizonte_min or self.horizonte_default),
                nivel_confianza=0.0,
                metodo_usado="Sin datos",
                mensaje_humano="aún sin histórico suficiente para pronosticar",
            )

        res = fc.predecir(float(pct_ocupado_actual),
                          horizonte_min=int(horizonte_min or self.horizonte_default))
        res.roi_key = key
        self._ultimo_resultado_por_roi[key] = res
        return res

    # ------------------------------------------------------------------ 5) helpers

    def forecast_all(self, cam_rois: Iterable[tuple[str, str]],
                     pct_ocupados: dict[tuple[str, str], float],
                     horizonte_min: int | None = None,
                     ) -> dict[str, ForecastResult]:
        """
        cam_rois: [(camera_id, roi_id), ...]
        pct_ocupados: {(camera_id, roi_id): 0..1}
        """
        out: dict[str, ForecastResult] = {}
        for cam, rid in cam_rois:
            ocup = pct_ocupados.get((cam, rid))
            if ocup is None:
                continue
            out[_roi_key(cam, rid)] = self.predecir_ahora(cam, rid, ocup, horizonte_min)
        return out

    def ultimo_resultado(self, cam_id: str, roi_id: str) -> ForecastResult | None:
        return self._ultimo_resultado_por_roi.get(_roi_key(cam_id, roi_id))

    def ranking_riesgo_ultima_semana(
        self,
        top_n: int = 5,
        umbral_min_alertas: int = 3,
    ) -> list[dict]:
        """
        Devuelve los N (cam, roi) con mayor riesgo esta semana:
        - pondera: número de veces que tiempo_hasta_vacio fue < umbral
        + tiempo medio hasta vacío (menos es peor).
        Se lee desde alerts.log + stock_history. NO guarda estado.
        """
        alerts_log = BASE / "data" / "alerts.log"
        filas_alerta: dict[str, int] = {}
        if alerts_log.exists():
            import csv
            with open(alerts_log, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                ahora = time.time()
                for r in reader:
                    try:
                        ts = time.mktime(time.strptime(r["timestamp_local"], "%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        continue
                    if (ahora - ts) > 7 * 24 * 3600:
                        continue
                    key = _roi_key(r["camera_id"], r["roi_id"])
                    filas_alerta[key] = filas_alerta.get(key, 0) + 1

        pares = self.logger.listar_camaras_y_rois()
        stats: list[dict] = []
        for cam, rid in pares:
            key = _roi_key(cam, rid)
            ts, ocup = self._cargar_serie(cam, rid)
            if len(ts) < 10:
                continue
            alertas = filas_alerta.get(key, 0)
            # Slope medio de ocupado (negativo = bajando):
            pendientes = []
            for i in range(1, len(ocup)):
                dt_min = (ts[i] - ts[i - 1]) / 60.0
                if dt_min > 0:
                    pendientes.append((ocup[i] - ocup[i - 1]) / dt_min)
            slope_medio = statistics.mean(pendientes) if pendientes else 0.0
            ocup_medio = statistics.mean(ocup)
            # Score riesgo:
            #   - slope < 0 => aumenta riesgo
            #   - ocup_medio bajo => aumenta riesgo
            #   - alertas >= umbral => aumenta riesgo
            score = (
                (-slope_medio) * 100.0            # 1 punto por %/min de bajada
                + (1.0 - ocup_medio) * 30.0       # penaliza ocup medio bajo
                + max(0, alertas - (umbral_min_alertas - 1)) * 20.0
            )
            stats.append({
                "camera_id": cam,
                "roi_id": rid,
                "muestras_hist": len(ts),
                "slope_medio_%_por_min": round(slope_medio * 100.0, 3),
                "ocupacion_media_%": round(ocup_medio * 100.0, 2),
                "alertas_7d": alertas,
                "score_riesgo": round(score, 2),
            })
        stats.sort(key=lambda s: s["score_riesgo"], reverse=True)
        return stats[:int(top_n)]
