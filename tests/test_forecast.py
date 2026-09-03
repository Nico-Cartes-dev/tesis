"""
Tests de toda la cadena de forecasting:
  - StockHistoryLogger (loggear, leer CSV, filtrar por ROI)
  - RollingSlopeForecaster (Nivel 1) — pendiente positiva = repone; pendiente
    negativa = consume; r2 del ajuste.
  - ProphetForecaster (Nivel 2) — al menos el fallback a RollingSlope debe
    funcionar sin necesidad de tener prophet ni statsmodels.
  - ForecastService (orquestación) — predecir_ahora con <2 datos, persistencia
    pickle no crashea.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from prediction.stock_history import (  # noqa: E402
    DEFAULT_CSV,
    CSV_COLUMNAS,
    MuestraStock,
    StockHistoryLogger,
)
from prediction.forecast_models import (  # noqa: E402
    ForecastResult,
    ProphetForecaster,
    RollingSlopeForecaster,
)
from prediction.forecast_service import ForecastService  # noqa: E402


def _ts_desde(minutos_antes: int, base_ts: float | None = None) -> float:
    base_ts = time.time() if base_ts is None else float(base_ts)
    return base_ts - minutos_antes * 60.0


class TestStockHistoryLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_stock_")
        self.csv_path = Path(self.tmp) / "h.csv"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_primer_log_crea_headers(self):
        lg = StockHistoryLogger(self.csv_path, intervalo_por_roi_seg=0.0)
        lg.loggear(MuestraStock("c1", "r1", 0.9, 0.1, 2, 0.05))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            headers = f.readline().strip().split(",")
        self.assertEqual(headers, CSV_COLUMNAS)

    def test_solo_loggea_cada_intervalo(self):
        lg = StockHistoryLogger(self.csv_path, intervalo_por_roi_seg=10.0)
        m = MuestraStock("c1", "r1", 0.9, 0.1, 2, 0.05)
        self.assertTrue(lg.loggear(m))
        self.assertFalse(lg.loggear(m))  # dentro del intervalo → no
        # Forzar intervalo con otra (cam, roi)
        self.assertTrue(lg.loggear(MuestraStock("c2", "r1", 0.9, 0.1, 2, 0.05)))

    def test_leer_roi_filtra_bien(self):
        lg = StockHistoryLogger(self.csv_path, intervalo_por_roi_seg=0.0)
        for i in range(5):
            lg.loggear(MuestraStock("c1", "r1", 1.0 - i * 0.1, i * 0.1, 2, 0.05))
        lg.loggear(MuestraStock("c1", "otra_roi", 0.5, 0.5, 0, 0.0))
        rows = lg.leer_roi("c1", "r1")
        self.assertEqual(len(rows), 5)


class TestRollingSlopeForecaster(unittest.TestCase):
    def setUp(self):
        self.ahora = time.time()

    def test_pendiente_negativa_predice_tiempo_hasta_cero(self):
        # 20 muestras: 1% por minuto de BAJA (consumo). Empezamos en 100% y
        # vamos bajando. Con slope=-1%/min deberíamos alcanzar 0 en ~100min.
        n = 20
        minutos_atras = list(range(n - 1, -1, -1))  # 19 min atras ... 0 (ahora)
        ts = [self.ahora - m * 60.0 for m in minutos_atras]
        # m=19 (muy atrás): ocup bajo (1.0 - 0.19) = 0.81
        # m=0  (ahora)  : ocup alto  = 1.00
        # PERO queremos slope negativo (ahora > pasado == POSITIVO == "repone").
        # Para slope negativo (está consumiendo): ahroa debe ser MENOR que pasado.
        # => invertimos la relación: y = 0.80 + 0.01 * (m_antiguedad)  → ahora < pasado
        ys = [1.0 - 0.01 * (n - 1 - m) for m in minutos_atras]
        # Check sanity: m más grande (más atrás) → más alto; ahora más bajo
        self.assertGreater(ys[0], ys[-1])
        fc = RollingSlopeForecaster(ventana_puntos=40)
        fc.ajustar(ts, ys)
        res = fc.predecir(pct_actual=ys[-1], horizonte_min=300)
        self.assertIsNotNone(res.tiempo_hasta_vacio_min)
        self.assertGreaterEqual(res.tiempo_hasta_vacio_min, 70.0)
        self.assertLessEqual(res.tiempo_hasta_vacio_min, 130.0)
        self.assertEqual(res.metodo_usado, "RollingSlope")

    def test_pendiente_positiva_no_predice_vacio(self):
        # Ahora SÍ está subiendo (reponiendo): el ocupado actual es el mayor.
        n = 5
        minutos_atras = list(range(n - 1, -1, -1))
        ts = [self.ahora - m * 60.0 for m in minutos_atras]
        ys = [0.4 + 0.05 * (n - 1 - m) for m in minutos_atras]
        self.assertLessEqual(ys[0], ys[-1])  # ahora >= pasado
        fc = RollingSlopeForecaster()
        fc.ajustar(ts, ys)
        res = fc.predecir(pct_actual=ys[-1], horizonte_min=120)
        self.assertIsNone(res.tiempo_hasta_vacio_min)


class TestProphetForecasterFallback(unittest.TestCase):
    def setUp(self):
        self.ahora = time.time()

    def test_fallback_funciona_sin_libs(self):
        # Con solo 10 puntos Prophet y SARIMA no entran → fallback a Rolling.
        # Generamos slope ~-1%/min (bajada de stock sostenida).
        n = 10
        minutos_atras = list(range(n - 1, -1, -1))
        ts = [self.ahora - m * 60.0 for m in minutos_atras]
        ys = [1.0 - 0.01 * (n - 1 - m) for m in minutos_atras]
        self.assertGreater(ys[0], ys[-1])
        fc = ProphetForecaster(min_filas_para_prophet=200)
        fc.ajustar(ts, ys)
        res = fc.predecir(pct_actual=ys[-1], horizonte_min=300)
        self.assertTrue(res.metodo_usado.startswith("RollingSlope"))
        self.assertIsNotNone(res.tiempo_hasta_vacio_min)


class TestForecastService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_fc_")
        self.csv = Path(self.tmp) / "hist.csv"
        self.ahora = time.time()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_predecir_ahora_sin_datos_no_crashea(self):
        svc = ForecastService(csv_path=self.csv)
        r = svc.predecir_ahora("c1", "r1", pct_ocupado_actual=0.8,
                               horizonte_min=120)
        self.assertEqual(r.roi_key, "c1::r1")
        self.assertIsNone(r.tiempo_hasta_vacio_min)
        self.assertEqual(r.metodo_usado, "Sin datos")

    def test_predecir_ahora_con_pocos_datos_usa_rolling(self):
        # Simular 10 min de historia bajando 1% / min (consumo sostenido).
        svc = ForecastService(csv_path=self.csv)
        svc.logger.intervalo = 0.0  # forzar escribir todas
        n = 10
        for i in range(n):
            antiguedad = n - 1 - i  # primera muestra más vieja
            t = self.ahora - antiguedad * 60.0
            # Queremos: MÁS VIEJO (antiguedad grande) = MÁS LLENO, AHORA = MÁS VACÍO
            #  => pendiente NEGATIVA (bajada de stock)
            ocup = 1.0 - 0.01 * (n - 1 - antiguedad)
            m = MuestraStock(
                "c1", "r1",
                pct_ocupado=ocup,
                pct_vacio=1.0 - ocup,
                num_detecciones=5,
                latencia_e2e_s=0.1,
                timestamp_unix=t,
            )
            svc.loggear_muestra(m)
        # Sanity check: el primer guardado (más viejo) debe ser el de mayor ocupado
        rows = svc.logger.leer_roi("c1", "r1")
        self.assertGreater(float(rows[0]["pct_ocupado"]), float(rows[-1]["pct_ocupado"]))
        # Recalcular y predecir
        svc.recalcular_para_roi("c1", "r1", forzar=True)
        ocup_actual = float(rows[-1]["pct_ocupado"])
        r = svc.predecir_ahora("c1", "r1", pct_ocupado_actual=ocup_actual,
                               horizonte_min=300, recalcular_si_hace_falta=False)
        self.assertIn("Rolling", r.metodo_usado)
        # El slope con 10 puntos al 1%/min debería dar ~ocup_actual / 0.01 min
        self.assertIsNotNone(r.tiempo_hasta_vacio_min)
        esperado = ocup_actual / 0.01
        self.assertGreaterEqual(r.tiempo_hasta_vacio_min, esperado * 0.6)
        self.assertLessEqual(r.tiempo_hasta_vacio_min, esperado * 1.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
