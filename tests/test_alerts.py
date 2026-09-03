"""
Tests para src/alerts/alert_manager.py
Cubre:
  - Umbral no superado → no alerta
  - Umbral superado → devuelve AlertaEvento y escribe en CSV
  - Cooldown por (camera, roi) → segunda alerta retorna None dentro del plazo
"""

from __future__ import annotations

import csv
import sys
import tempfile
import time
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from alerts.alert_manager import AlertManager  # noqa: E402


class TestAlertManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
        self._tmp.close()
        self.log_path = Path(self._tmp.name)

    def tearDown(self):
        try:
            self.log_path.unlink()
        except FileNotFoundError:
            pass

    def test_umbral_no_superado_no_alerta(self):
        """Si pct_vacio < umbral, alertar retorna None y no escribe."""
        am = AlertManager(log_path=str(self.log_path), cooldown_por_roi_seg=0.0)
        evento = am.alertar(
            camera_id="c1", roi_id="r1",
            pct_vacio=0.10, pct_umbral=0.35,
        )
        self.assertIsNone(evento)
        # Cerramos para que se flushie el CSV y leemos líneas
        am.close()
        contenido = self.log_path.read_text(encoding="utf-8").splitlines()
        # Solo header (0 líneas extras)
        lineas_datos = [l for l in contenido if l and not l.startswith("timestamp_utc")]
        self.assertEqual(len(lineas_datos), 0)

    def test_umbral_superado_alerta_una_vez(self):
        am = AlertManager(log_path=str(self.log_path), cooldown_por_roi_seg=0.0)
        evento = am.alertar(
            camera_id="c1", roi_id="r1",
            pct_vacio=0.50, pct_umbral=0.35,
            metadatos={"foo": "bar"},
        )
        self.assertIsNotNone(evento)
        self.assertEqual(evento.camera_id, "c1")
        self.assertEqual(evento.roi_id, "r1")
        self.assertAlmostEqual(evento.pct_vacio, 0.50)
        self.assertIn("ALERTA STOCK", evento.mensaje)
        am.close()

        filas = list(csv.reader(self.log_path.read_text(encoding="utf-8").splitlines()))
        # header + 1 dato
        self.assertEqual(len(filas), 2)
        self.assertIn("c1", filas[1])
        self.assertIn("r1", filas[1])

    def test_cooldown_evita_alertas_duplicadas(self):
        am = AlertManager(log_path=str(self.log_path), cooldown_por_roi_seg=10_000)
        e1 = am.alertar("c1", "r1", pct_vacio=0.50, pct_umbral=0.35)
        self.assertIsNotNone(e1)
        e2 = am.alertar("c1", "r1", pct_vacio=0.50, pct_umbral=0.35)
        self.assertIsNone(e2)
        am.close()
        filas = list(csv.reader(self.log_path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(len(filas), 2, "debe haber solo 1 fila de datos + 1 header")

    def test_cooldown_expira_y_vuelve_a_alertar(self):
        am = AlertManager(log_path=str(self.log_path), cooldown_por_roi_seg=0.05)
        e1 = am.alertar("c1", "r1", pct_vacio=0.50, pct_umbral=0.35)
        self.assertIsNotNone(e1)
        time.sleep(0.08)
        e2 = am.alertar("c1", "r1", pct_vacio=0.50, pct_umbral=0.35)
        self.assertIsNotNone(e2)
        am.close()
        filas = list(csv.reader(self.log_path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(len(filas), 3, "debe haber 2 filas de datos + header")


if __name__ == "__main__":
    unittest.main()
