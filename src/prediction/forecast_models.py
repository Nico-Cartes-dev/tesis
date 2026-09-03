"""
Modelos de forecasting (predicción) para estimar CUÁNDO se acabará el stock
de una ROI dada, usando el histórico de `data/stock_history.csv`.

DOS CAPAS (API común, `ForecasterABC`):
  1) RollingSlopeForecaster  → extrapola con slope lineal últimos N puntos.
     SIN entrenamiento, sirve desde el minuto 1.
  2) ProphetForecaster       → usa prophet de Meta si está instalado;
     si no, fallback a statsmodels SARIMA (o si ese también falta, vuelve a
     RollingSlope, pero lo marca con baja confianza).
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ---------------- Interfaz común ----------------

@dataclass
class ForecastResult:
    """Respuesta estándar de cualquier forecaster."""
    roi_key: str                             # "camara_02::repisa_yogures"
    pct_ocupado_actual: float                # 0..1
    pct_vacio_actual: float                  # 0..1
    tiempo_hasta_vacio_min: float | None     # None = "no se acaba" / tendencia plana
    fecha_estimada_vacio: dt.datetime | None # None = "no se acaba"
    horizonte_minutos: int                   # cuánto futuro se predijo
    nivel_confianza: float                   # 0..1 (RollingSlope usa R^2 del ajuste)
    metodo_usado: str                        # "RollingSlope" | "Prophet" | "SARIMA"
    mensaje_humano: str = ""                 # ej: "se acaba en ~12 min"


class ForecasterABC:
    def ajustar(self, timestamps_unix: Sequence[float],
                pct_ocupados: Sequence[float]) -> None:
        raise NotImplementedError

    def predecir(self, pct_actual: float,
                 horizonte_min: int = 180) -> ForecastResult:
        raise NotImplementedError


# ---------------- Helpers generales ----------------

def _filtrar_serie(ts: Sequence[float], vals: Sequence[float]):
    """Elimina NaNs y ordena por tiempo."""
    pares = sorted([(float(t), float(v)) for t, v in zip(ts, vals)
                    if not math.isnan(float(t)) and not math.isnan(float(v))])
    if not pares:
        return [], []
    return zip(*pares)


def _regresion_lineal(x: np.ndarray, y: np.ndarray):
    """Ajuste y = m*x + b por mínimos cuadrados. Devuelve (m, b, r2)."""
    n = len(x)
    if n < 2:
        return 0.0, float(np.mean(y)) if n else 0.0, 0.0
    mx, my = x.mean(), y.mean()
    num = np.sum((x - mx) * (y - my))
    den = np.sum((x - mx) ** 2)
    if den < 1e-12:
        return 0.0, my, 0.0
    m = num / den
    b = my - m * mx
    ss_res = np.sum((y - (m * x + b)) ** 2)
    ss_tot = np.sum((y - my) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return float(m), float(b), float(r2)


def _fecha_en(ts_actual_s: float, suma_min: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts_actual_s + suma_min * 60.0)


# =====================================================================
# NIVEL 1 — Rolling slope (SIN entrenamiento, sin dependencias extra)
# =====================================================================

class RollingSlopeForecaster(ForecasterABC):
    """
    Extrapola la tendencia del % ocupado en los últimos `ventana_puntos` puntos.
    Sirve a partir de tener 2-3 muestras (2-3 minutos de logging).
    """

    def __init__(self, ventana_puntos: int = 20, umbral_pendiente_negativa_min: float = 1e-6):
        self.ventana = max(2, int(ventana_puntos))
        self.umbral_pendiente = abs(umbral_pendiente_negativa_min)
        self.ts: np.ndarray | None = None
        self.ocup: np.ndarray | None = None
        self._r2 = 0.0

    # ---- entrenamiento / ajuste ----
    def ajustar(self, timestamps_unix: Sequence[float],
                pct_ocupados: Sequence[float]) -> None:
        ts, ocup = _filtrar_serie(timestamps_unix, pct_ocupados)
        ts, ocup = np.asarray(list(ts), dtype=float), np.asarray(list(ocup), dtype=float)
        if len(ts) > self.ventana:
            ts = ts[-self.ventana:]
            ocup = ocup[-self.ventana:]
        self.ts = ts
        self.ocup = ocup

    # ---- predicción ----
    def predecir(self, pct_actual: float,
                 horizonte_min: int = 180) -> ForecastResult:
        roi_key = "generic"
        assert self.ts is not None and self.ocup is not None, "Ajustar() antes de predecir()"

        ts_min = (self.ts - self.ts[-1]) / 60.0  # minutos relativos = [.. , -2, -1, 0]
        y = self.ocup
        m, b, r2 = _regresion_lineal(ts_min, y)
        self._r2 = r2
        ocup_actual = float(pct_actual if pct_actual is not None else float(y[-1]))
        vacio_actual = 1.0 - ocup_actual

        # Pendiente MUY pequeña o POSITIVA (reponiendo) → no se acaba a la vista
        if m >= -self.umbral_pendiente:
            return ForecastResult(
                roi_key=roi_key,
                pct_ocupado_actual=ocup_actual,
                pct_vacio_actual=vacio_actual,
                tiempo_hasta_vacio_min=None,
                fecha_estimada_vacio=None,
                horizonte_minutos=horizonte_min,
                nivel_confianza=max(0.0, min(1.0, r2)),
                metodo_usado="RollingSlope",
                mensaje_humano="estable / repuestos en curso, no se proyecta vacío",
            )

        # Resolvemos:  m * t + b = 0  → t = -b / m  (usamos ocup_actual como ref)
        # Pero usamos la proyección desde el punto actual para mayor precisión:
        # ocup(t) = ocup_actual + m * t  → ocup(t) = 0 → t = -ocup_actual / m
        t_min = -ocup_actual / m
        # Clampeamos al horizonte (si > horizonte, decimos "no a la vista")
        if t_min > horizonte_min:
            return ForecastResult(
                roi_key=roi_key,
                pct_ocupado_actual=ocup_actual,
                pct_vacio_actual=vacio_actual,
                tiempo_hasta_vacio_min=float(t_min),
                fecha_estimada_vacio=_fecha_en(self.ts[-1], t_min),
                horizonte_minutos=horizonte_min,
                nivel_confianza=max(0.0, min(1.0, r2)) * 0.6,  # descuento por estar lejos
                metodo_usado="RollingSlope",
                mensaje_humano=f"se vacía en ~{t_min:.0f} min (fuera de horizonte {horizonte_min} min)",
            )

        return ForecastResult(
            roi_key=roi_key,
            pct_ocupado_actual=ocup_actual,
            pct_vacio_actual=vacio_actual,
            tiempo_hasta_vacio_min=float(t_min),
            fecha_estimada_vacio=_fecha_en(self.ts[-1], t_min),
            horizonte_minutos=horizonte_min,
            nivel_confianza=max(0.0, min(1.0, r2)),
            metodo_usado="RollingSlope",
            mensaje_humano=f"se vacía aproximadamente en {t_min:.0f} minutos",
        )


# =====================================================================
# NIVEL 2 — Prophet (Meta) / fallback SARIMA
# =====================================================================

def _tiene_prophet() -> bool:
    try:
        import prophet  # noqa: F401
        return True
    except Exception:
        return False


def _tiene_sarima() -> bool:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
        return True
    except Exception:
        return False


class ProphetForecaster(ForecasterABC):
    """
    Capa 2: usa Prophet de Meta cuando hay suficientes datos (>= 200 filas)
    y la librería está disponible. Si Prophet falta, prueba SARIMA.
    Si nada está disponible, construye un RollingSlope internamente y
    marca `metodo_usado = "RollingSlope(Fallback)"` con confianza reducida.
    """

    def __init__(self, min_filas_para_prophet: int = 200):
        self.min_filas = int(min_filas_para_prophet)
        self._modelo = None          # prophet / SARIMAXResults / RollingSlopeForecaster
        self._ultimos_t: list[float] = []
        self._ultimos_y: list[float] = []
        self._metodo: str = ""

    # ---- ajuste ----
    def ajustar(self, timestamps_unix: Sequence[float],
                pct_ocupados: Sequence[float]) -> None:
        ts, ocup = _filtrar_serie(timestamps_unix, pct_ocupados)
        ts = list(ts); ocup = list(ocup)
        self._ultimos_t = ts
        self._ultimos_y = ocup
        n = len(ts)

        # Camino 1: Prophet
        if _tiene_prophet() and n >= self.min_filas:
            import pandas as pd
            from prophet import Prophet
            df = pd.DataFrame({
                "ds": [dt.datetime.fromtimestamp(t) for t in ts],
                "y": ocup,
            })
            m = Prophet(daily_seasonality=True, weekly_seasonality=True,
                        yearly_seasonality=False)
            try:
                m.fit(df, algorithm="LBFGS")
            except Exception:
                m.fit(df)
            self._modelo = m
            self._metodo = "Prophet"
            return

        # Camino 2: SARIMA
        if _tiene_sarima() and n >= max(30, self.min_filas // 3):
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            try:
                mod = SARIMAX(ocup, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                self._modelo = res
                self._metodo = "SARIMA"
                return
            except Exception:
                pass

        # Camino 3: fallback
        fb = RollingSlopeForecaster(ventana_puntos=min(30, max(3, n)))
        fb.ajustar(ts, ocup)
        self._modelo = fb
        self._metodo = "RollingSlope(Fallback)"

    # ---- predicción ----
    def predecir(self, pct_actual: float,
                 horizonte_min: int = 180) -> ForecastResult:
        n = len(self._ultimos_t)
        ts_last = self._ultimos_t[-1] if n else time.time()
        y_last = float(self._ultimos_y[-1]) if n else 0.0
        ocup_actual = float(pct_actual if pct_actual is not None else y_last)
        vacio_actual = 1.0 - ocup_actual
        base = ForecastResult(
            roi_key="generic", pct_ocupado_actual=ocup_actual,
            pct_vacio_actual=vacio_actual, tiempo_hasta_vacio_min=None,
            fecha_estimada_vacio=None, horizonte_minutos=horizonte_min,
            nivel_confianza=0.0, metodo_usado=self._metodo or "?",
            mensaje_humano="sin datos suficientes",
        )

        # Prophet
        if self._metodo == "Prophet":
            import pandas as pd
            futuro = self._modelo.make_future_dataframe(periods=horizonte_min, freq="min")
            fc = self._modelo.predict(futuro)
            serie_yh = fc["yhat"].values
            idx_cruce = next((i for i, yh in enumerate(serie_yh[-horizonte_min:], start=len(serie_yh)-horizonte_min)
                              if yh <= 0), None)
            base.nivel_confianza = 0.85
            if idx_cruce is None:
                base.mensaje_humano = f"Prophet: no alcanza 0 en {horizonte_min} min"
                base.tiempo_hasta_vacio_min = None
            else:
                delta_min = (fc["ds"].iloc[idx_cruce] - pd.Timestamp.fromtimestamp(ts_last)).total_seconds() / 60.0
                if delta_min < 0:
                    delta_min = 0
                base.tiempo_hasta_vacio_min = float(delta_min)
                base.fecha_estimada_vacio = _fecha_en(ts_last, delta_min)
                base.mensaje_humano = f"Prophet: se vacía ~ {delta_min:.0f} min"
            return base

        # SARIMA
        if self._metodo == "SARIMA":
            try:
                fc = self._modelo.get_forecast(steps=horizonte_min).predicted_mean
            except Exception:
                base.mensaje_humano = "SARIMA no pudo predecir"
                return base
            idx = next((i for i, y in enumerate(fc) if y <= 0), None)
            base.nivel_confianza = 0.7
            if idx is None:
                base.mensaje_humano = f"SARIMA: no alcanza 0 en {horizonte_min} min"
                return base
            base.tiempo_hasta_vacio_min = float(idx + 1)
            base.fecha_estimada_vacio = _fecha_en(ts_last, idx + 1)
            base.mensaje_humano = f"SARIMA: se vacía ~ {idx+1:.0f} min"
            return base

        # Fallback a RollingSlope
        if isinstance(self._modelo, RollingSlopeForecaster):
            r = self._modelo.predecir(ocup_actual, horizonte_min)
            base.tiempo_hasta_vacio_min = r.tiempo_hasta_vacio_min
            base.fecha_estimada_vacio = r.fecha_estimada_vacio
            base.nivel_confianza = max(0.0, r.nivel_confianza - 0.15)
            base.mensaje_humano = f"{self._metodo}: {r.mensaje_humano}"
            return base

        return base
