"""
Generador de resúmenes en lenguaje natural para las alertas, usando un LLM.

CORRESPONDE al rol del LLM definido en kickoff-notes.md:
  > Tomar la alerta cruda y transformarla en un mensaje en español para
    encargados / repositores, con contexto útil.

Modo de uso sin API KEY (OFFLINE / STUB):
  - Simula una respuesta basada en plantillas. Sirve para integrarlo hoy
    en el pipeline sin tener que poner credenciales reales.

Modo REAL (cuando tengas tu API key):
  - Instala `pip install openai` (u otro proveedor).
  - Setea la variable de entorno:  $env:OPENAI_API_KEY = "sk-..."
  - Cambia `cliente=None` por una instancia real.

Este módulo NO toca la ventana OpenCV ni el modelo YOLO. Se integra con
AlertManager a través de `extra_callback`. Ver ejemplo al final del archivo.
"""

from __future__ import annotations

import csv
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

if False:  # importo solo para type-check, no requiere la lib instalada
    from alerts.alert_manager import AlertaEvento


BASE = Path(__file__).resolve().parent.parent.parent


# -------------------- helpers de plantilla --------------------

PROMPT_SISTEMA = """Eres un asistente de operaciones de supermercado en español.
Tu tarea: a partir de una alerta técnica sobre stock faltante en una góndola,
generar UN SOLO mensaje breve (4-6 renglones) para el encargado / repositor.
Incluye:
  - qué repisa / cámara,
  - cuánto % de vacío (nivel),
  - tiempo que lleva así (si aplica),
  - una recomendación práctica de reposición
  - una sugerencia de horario basada en un historial breve (si hay datos).
No usas jerga de CV ni hablas de bounding boxes. Sé claro y accionable."""


PLANTILLAS_OFFLINE = [
    "⚠️ Alerta de stock — {nombre_cam} (ROI {roi})\n"
    "  Nivel actual de vacío: {vacio:.0f}% (umbral {umbral:.0f}%)\n"
    "  Lleva {t_seg:.0f}s por encima del umbral.\n"
    "  => RECOMENDACIÓN: priorizar reposición lo antes posible.\n"
    "  Sugerencia de reposición: {sugerencia} unidades (estimado por espacio libre).",

    "📉 Stock bajo detectado en cámara {cam_id}, repisa '{roi}'.\n"
    "  Vacío: {vacio:.0f}% | Ocupado: {ocup:.0f}%.\n"
    "  Hora local: {hora}.\n"
    "  Acción: acercar stock al pasillo para rellenar huecos.",

    "🛒 Atención pasillo — {cam} / {roi}.\n"
    "  {pct_vacio_str}% sin producto (umbral {um:.0f}%).\n"
    "  Estado sostenido: {estado}.\n"
    "  Típicamente se rellena mejor antes de la franja: {franja}.",
]


def _franja_horaria_sugerida(hora_local: datetime) -> str:
    h = hora_local.hour
    if 7 <= h < 13:
        return "12:00 (antes del almuerzo)"
    if 13 <= h < 19:
        return "18:00 (antes del pico de la tarde)"
    return "21:00 (antes de cierre)"


def _estimacion_unidades(pct_vacio: float, area_roi_px: int | None) -> int:
    # Si no tenemos área, aproximación por % (heurística por repisa típica 12u)
    capacidad_aproximada = 12
    if area_roi_px and area_roi_px > 0:
        # asumimos ~1 producto = 4000 px; clamp a [6, 24]
        capacidad_aproximada = max(6, min(24, int(round(area_roi_px / 4000))))
    return int(round(capacidad_aproximada * max(0.0, min(1.0, pct_vacio))))


# -------------------- clase principal --------------------

@dataclass
class HistoricoAlerta:
    roi_id: str
    hora_local: datetime
    pct_vacio: float


class LLMResumidor:
    """Convierte AlertaEvento en texto en español para humano.

    Con cliente LLM real: usa llamadas a API (OpenAI-compatible).
    Sin cliente: usa plantillas offline + historial reciente.
    """

    def __init__(
        self,
        cliente=None,          # instancia openai.AsyncOpenAI / OpenAI / compatible
        modelo: str = "gpt-4o-mini",
        usar_offline_si_falla: bool = True,
        historico_path: Path | str | None = None,
    ):
        self.cliente = cliente
        self.modelo = modelo
        self.usar_offline = usar_offline_si_falla
        self.historico_path = Path(historico_path) if historico_path else \
            BASE / "data" / "alerts_historial.csv"
        self.historico_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_historico_csv()

    # ------------------------------------------------------------------ CSV
    def _init_historico_csv(self) -> None:
        if self.historico_path.exists() and self.historico_path.stat().st_size > 0:
            return
        with open(self.historico_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_local", "camera_id", "roi_id",
                        "pct_vacio", "pct_umbral", "area_roi_px"])

    def _guardar_historico(self, evento, area_roi_px: int | None) -> None:
        with open(self.historico_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.fromtimestamp(evento.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                evento.camera_id,
                evento.roi_id,
                f"{evento.pct_vacio:.4f}",
                f"{evento.pct_umbral:.4f}",
                "" if area_roi_px is None else area_roi_px,
            ])

    def _leer_historico_roi(self, roi_id: str, ultimas_filas: int = 30) -> list[HistoricoAlerta]:
        if not self.historico_path.exists():
            return []
        out: list[HistoricoAlerta] = []
        with open(self.historico_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            filas = [r for r in reader if r["roi_id"] == roi_id][-ultimas_filas:]
        for r in filas:
            try:
                out.append(HistoricoAlerta(
                    roi_id=r["roi_id"],
                    hora_local=datetime.strptime(r["timestamp_local"], "%Y-%m-%d %H:%M:%S"),
                    pct_vacio=float(r["pct_vacio"]),
                ))
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------ gen
    def resumir(
        self,
        evento,  # AlertaEvento
        segundos_sostenidos: float | None = None,
    ) -> str:
        """Genera el texto para el humano. Nunca crashea si falla API."""
        area_roi_px = (evento.metadatos or {}).get("area_roi_px")
        pct_ocup = 1.0 - evento.pct_vacio
        self._guardar_historico(evento, area_roi_px)

        # Contexto historial (para ambos modos, online y offline)
        hist = self._leer_historico_roi(evento.roi_id)

        # Detectamos si viene de Forecast (PROACTIVA) o de umbral (REACTIVA)
        md = evento.metadatos or {}
        tipo_alerta = md.get("tipo", "REACTIVA_umbral_vacio")
        t_min = md.get("forecast_min_hasta_vacio")       # minutos hasta vacío (float)
        eta_local = md.get("forecast_fecha_local")       # str "YYYY-MM-DD HH:MM:SS" (puede ser None)
        metodo_fc = md.get("forecast_metodo", None)
        conf_fc = md.get("forecast_conf", None)

        if self.cliente is not None:
            try:
                return self._llamar_api(
                    evento, segundos_sostenidos, hist, area_roi_px, pct_ocup,
                    tipo_alerta=tipo_alerta, t_min=t_min,
                    eta_local=eta_local, metodo_fc=metodo_fc, conf_fc=conf_fc,
                )
            except Exception as e:
                if not self.usar_offline:
                    raise RuntimeError(f"LLM falló: {e}") from e
        # Modo offline (plantillas + aleatoriedad leve)
        return self._plantilla_offline(
            evento, segundos_sostenidos, hist, area_roi_px, pct_ocup,
            tipo_alerta=tipo_alerta, t_min=t_min,
            eta_local=eta_local, metodo_fc=metodo_fc, conf_fc=conf_fc,
        )

    # ------------------------------------------------------------------ impl offline
    def _plantilla_offline(
        self, evento, t_seg, hist, area_roi, pct_ocup,
        tipo_alerta, t_min, eta_local, metodo_fc, conf_fc,
    ) -> str:
        tpl = random.choice(PLANTILLAS_OFFLINE)
        ahora = datetime.fromtimestamp(evento.timestamp)
        hist_vacios = [h.pct_vacio for h in hist] or [evento.pct_vacio]
        estado = (
            f"{'habitual' if statistics.mean(hist_vacios) >= evento.pct_umbral else 'nuevo'}"
            f" — {len(hist)} registros en histórico"
        )
        t_seg = t_seg or 0.0
        base = tpl.format(
            cam=evento.camera_id, cam_id=evento.camera_id,
            nombre_cam=evento.camera_id,
            roi=evento.roi_id,
            vacio=evento.pct_vacio * 100,
            ocup=pct_ocup * 100,
            pct_vacio_str=f"{evento.pct_vacio * 100:.0f}",
            umbral=evento.pct_umbral * 100, um=evento.pct_umbral * 100,
            t_seg=float(t_seg),
            hora=ahora.strftime("%H:%M"),
            sugerencia=_estimacion_unidades(evento.pct_vacio, area_roi),
            estado=estado,
            franja=_franja_horaria_sugerida(ahora),
        )
        # ------------------------------------------------------------------ FORECAST extra
        if tipo_alerta == "PROACTIVA_forecast":
            t_min_str = "-" if t_min is None else f"{t_min:.0f} min"
            eta_str = eta_local or "-"
            conf_str = "-" if conf_fc is None else f"{float(conf_fc)*100:.0f}%"
            fc_header = (
                "\n\n🔮 [ALERTA PROACTIVA / TENDENCIA] — "
                f"el modelo ({metodo_fc or 'RollingSlope'}) "
                f"estima que se acabará en ~{t_min_str} (ETA {eta_str}, conf={conf_str})."
            )
            if t_min is not None and t_min <= 10:
                urgencia = "\n  ⚡ URGENTE: menos de 10 min para quedar vacío. Reponer INMEDIATAMENTE."
            elif t_min is not None and t_min <= 30:
                urgencia = "\n  ⏳ Corto plazo (≤30 min). Preparar reposición ahora."
            elif t_min is not None and t_min <= 60:
                urgencia = "\n  📋 Medio plazo (≤60 min). Agendar reposición."
            else:
                urgencia = "\n  📝 Largo plazo. Registrar para planificación."
            base = fc_header + urgencia + "\n---\n" + base
        return base

    # ------------------------------------------------------------------ impl online
    def _llamar_api(
        self, evento, t_seg, hist, area_roi, pct_ocup,
        tipo_alerta, t_min, eta_local, metodo_fc, conf_fc,
    ) -> str:
        ahora = datetime.fromtimestamp(evento.timestamp)
        datos_tec = {
            "camera_id": evento.camera_id,
            "roi_id": evento.roi_id,
            "pct_vacio": round(evento.pct_vacio, 4),
            "pct_ocupado": round(pct_ocup, 4),
            "pct_umbral": round(evento.pct_umbral, 4),
            "area_roi_px": area_roi,
            "segundos_sostenidos": None if t_seg is None else round(float(t_seg), 2),
            "timestamp_local": ahora.strftime("%Y-%m-%d %H:%M:%S"),
            "ultimas_alertas_mismo_ROI": [
                {"hora": h.hora_local.strftime("%H:%M"),
                 "pct_vacio": round(h.pct_vacio, 3)}
                for h in hist[-8:]
            ],
            "sugerencia_unidades": _estimacion_unidades(evento.pct_vacio, area_roi),
            "franja_horaria_sugerida": _franja_horaria_sugerida(ahora),
            "metadatos_extras": evento.metadatos,
            "forecast": {
                "tipo_alerta": tipo_alerta,
                "minutos_hasta_vacio": t_min,
                "eta_local": eta_local,
                "metodo": metodo_fc,
                "confianza": conf_fc,
            },
        }
        prompt_sistema = PROMPT_SISTEMA
        if tipo_alerta == "PROACTIVA_forecast":
            prompt_sistema = (
                PROMPT_SISTEMA
                + "\nIMPORTANTE: esta es una ALERTA PROACTIVA (basada en forecast de tendencia)."
                + "\nDestacá CUÁNDO (tiempo + hora) se estima que se acabará el producto y la URGENCIA."
            )
        user_msg = (
            "DATOS CRUDOS DE LA ALERTA (JSON):\n"
            + __import__("json").dumps(datos_tec, ensure_ascii=False, indent=2)
        )
        try:
            resp = self.cliente.chat.completions.create(
                model=self.modelo,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover - runtime
            raise RuntimeError(f"Error llamada API LLM: {e}") from e


# -------------------- callback para AlertManager --------------------

def armar_callback_resumidor(resumidor: LLMResumidor,
                             destino_texto: Path | None = None):
    """Devuelve un callback listo para AlertManager(extra_callback=...).
    Aparte de generar el resumen, lo escribe a un .txt para dejar traza.
    """
    if destino_texto is None:
        destino_texto = BASE / "data" / "alerts_resumen.txt"
    destino_texto.parent.mkdir(parents=True, exist_ok=True)

    def cb(evento):
        t_sost = None
        md = evento.metadatos or {}
        # Latencia E2E y otros datos que detect.py colgó en metadatos
        resumen = resumidor.resumir(evento, segundos_sostenidos=t_sost)
        with open(destino_texto, "a", encoding="utf-8") as f:
            f.write("=" * 78 + "\n")
            f.write(datetime.fromtimestamp(evento.timestamp).strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(resumen + "\n\n")
        print("\n" + resumen + "\n")
        return resumen
    return cb


# -------------------- demo / smoke test --------------------

if __name__ == "__main__":
    # 1) Crear resumidor OFFLINE
    resumidor = LLMResumidor(cliente=None)
    # 2) Construir un AlertaEvento fake
    sys.path.insert(0, str(BASE / "src"))
    from alerts.alert_manager import AlertaEvento  # noqa: E402

    ev = AlertaEvento(
        timestamp=time.time(),
        camera_id="camara_02",
        roi_id="repisa_yogures",
        pct_vacio=0.58,
        pct_umbral=0.35,
        metadatos={"area_roi_px": 42000, "pct_ocupado": 0.42},
    )
    texto = resumidor.resumir(ev, segundos_sostenidos=240)
    print("=" * 60)
    print("DEMO resumen LLM (modo OFFLINE):\n")
    print(texto)
    print("=" * 60)
