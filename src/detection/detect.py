"""
Demo interactiva de detección de stock. Abre una ventana con:
  - Imagen / frame con ROIs dibujados
  - Bounding boxes de productos detectados (verde)
  - Por cada ROI: cálculo de % ocupado vs % vacío + alerta (rojo) si corresponde
  - Forecast de cuando se acaba cada repisa (naranja si es anticipada)

Pipeline por frame (según kickoff):
  Frame crudo  →  Preprocesamiento (denoise + CLAHE + gamma opc)  →  YOLO predict  →
  % stock por ROI  →  Criterio sostenido  →  Alerta REACTIVA (umbral)
                      +  Forecast (RollingSlope / Prophet / SARIMA)
                      →  Alerta PROACTIVA anticipada si se acaba < X min

Tres modos de entrada (elegí uno):
  1) --folder <carpeta>  -> navega imágenes locales con trackbar (ideal para probar dataset)
  2) --image <archivo>   -> una sola imagen
  3) --video <archivo|0> -> video o cámara web (0 = cámara default)

Modelo:
  - --weights: path al .pt entrenado (ej: models/runs/stock_detector_v1/weights/best.pt)
    Si todavía no lo entrenaste, podés usar el script src/utils/visualize_dataset.py
    para ver el ground-truth (labels reales sin inferencia).

Uso:
  # Ver inferencia sobre imágenes de val con tu modelo entrenado
  python src/detection/detect.py --weights models/runs/stock_detector_v1/weights/best.pt \
         --folder data/processed/images/val

  # Ver una imagen sola + ROI de una cámara
  python src/detection/detect.py --weights best.pt --image data/raw/ejemplo.jpg \
         --camera camara_01

  # Ver cámara web (si tuvieras pesos entrenados) — empieza a guardar data/stock_history.csv
  python src/detection/detect.py --weights best.pt --video 0 --camera camara_01
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

from preprocessing.roi_handler import ROIHandler  # noqa: E402
from preprocessing.image_enhancer import ImageEnhancer, PreprocessConfig  # noqa: E402
from alerts.alert_manager import AlertManager  # noqa: E402
from prediction.stock_history import MuestraStock  # noqa: E402
from prediction.forecast_service import ForecastService  # noqa: E402
from prediction.forecast_models import ForecastResult  # noqa: E402

WIN = "Detección de stock - Demo"

CLASSES = {0: ("vacio", (0, 0, 255)),
           1: ("producto", (0, 200, 0))}


def color_segun_tamano(pct_area_roi: float, base_cls: int) -> tuple[int, int, int]:
    """
    Devuelve color para bounding boxes de huecos/productos según tamaño relativo al ROI.
    - Huecos (cls=0):
        < 5% ROI     : verde (muy chico, ruido leve / ok)
        5% - 20%     : amarillo (chico)
        20% - 35%    : naranja (medio)
        > 35%        : rojo (grande, supera umbral de alerta)
    - Productos (cls=1):
        verde siempre.
    """
    if base_cls != 0:
        return CLASSES[1][1]
    if pct_area_roi < 0.05:
        return (0, 200, 0)
    if pct_area_roi < 0.20:
        return (0, 200, 255)
    if pct_area_roi < 0.35:
        return (0, 120, 255)
    return (0, 0, 255)


def enhancer_desde_config(cfg_global: dict | None) -> ImageEnhancer:
    """Construye ImageEnhancer a partir de la sección 'preprocesamiento' del yaml."""
    if not cfg_global:
        return ImageEnhancer()
    pre = cfg_global.get("preprocesamiento", {}) or {}
    den = pre.get("denoise", {}) or {}
    cl = pre.get("clahe", {}) or {}
    gm = pre.get("gamma", {}) or {}
    tgs = cl.get("tile_grid_size") or [8, 8]
    return ImageEnhancer(PreprocessConfig(
        denoise_activo=bool(den.get("activo", True)),
        denoise_fuerza=int(den.get("fuerza", 7)),
        clahe_activo=bool(cl.get("activo", True)),
        clahe_clip_limit=float(cl.get("clip_limit", 2.0)),
        clahe_tile_grid_size=(int(tgs[0]), int(tgs[1])),
        gamma_activo=bool(gm.get("activo", False)),
        gamma_valor=float(gm.get("gamma_valor", 1.0)),
    ))


def cargar_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def interseccion_sobre_area_roi(box, roi_mask, alto, ancho):
    """Calcula el área (en px) de un bbox que cae DENTRO de la máscara de un ROI."""
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(ancho, int(x2)); y2 = min(alto, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0
    submask = roi_mask[y1:y2, x1:x2]
    return int(cv2.countNonZero(submask))


def mask_bbox_en_roi(img_h, img_w, x1, y1, x2, y2, mask_roi):
    """Crea una máscara binaria del bbox, recortada al ROI (devuelve los píxeles cubiertos)."""
    sub = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.rectangle(sub, (max(0, int(x1)), max(0, int(y1))),
                  (min(img_w, int(x2)), min(img_h, int(y2))), 255, -1)
    if mask_roi is not None:
        return cv2.bitwise_and(sub, mask_roi)
    return sub


def calcular_ocupacion(img_h, img_w, detecciones, roi_poly_px,
                       modo: str = "productos",
                       area_min_rel_hueco: float = 0.01,
                       area_min_rel_producto: float = 0.005,
                       area_roi=None, mask_roi=None):
    """
    Calcula pct_ocupado / pct_vacío según el `modo` de stock elegido.

    Modos:
      - 'productos' (clásico): YOLO detecta productos (cls==1). %ocupado = union(productos)/area_roi
      - 'huecos' (recomendado por tesis): YOLO detecta huecos (cls==0). %vacío = union(huecos)/area_roi
      - 'mixto' (más robusto): combina ambos = max( area_huecos , 1 − area_{productos ∪ huecos} ) / area_roi
    """
    if mask_roi is None:
        mask_roi = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [roi_poly_px], 255)
    if area_roi is None:
        area_roi = int(cv2.countNonZero(mask_roi))
    if area_roi == 0:
        return 0.0, 0.0, area_roi, []

    modo = (modo or "productos").lower()
    min_h = max(0.0, float(area_min_rel_hueco))
    min_p = max(0.0, float(area_min_rel_producto))

    # Máscaras separadas y lista de detecciones filtradas para dibujo posterior
    mask_huecos = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_prods = np.zeros((img_h, img_w), dtype=np.uint8)
    dets_validadas = []  # (x1, y1, x2, y2, cls, conf, area_en_roi_rel, area_en_roi_px, color)

    for x1, y1, x2, y2, cls, conf in detecciones:
        area_en_roi_px = interseccion_sobre_area_roi(
            (x1, y1, x2, y2), mask_roi, img_h, img_w)
        if area_en_roi_px == 0:
            continue
        area_rel = area_en_roi_px / area_roi
        if cls == 0 and area_rel < min_h:
            continue  # hueco muy chico → chamusca
        if cls == 1 and area_rel < min_p:
            continue  # producto muy chico → ruido

        color = color_segun_tamano(area_rel, base_cls=int(cls))
        dets_validadas.append(
            (float(x1), float(y1), float(x2), float(y2),
             int(cls), float(conf), float(area_rel),
             int(area_en_roi_px), color))

        if cls == 0:
            mask_huecos = cv2.bitwise_or(
                mask_huecos,
                mask_bbox_en_roi(img_h, img_w, x1, y1, x2, y2, mask_roi))
        elif cls == 1:
            mask_prods = cv2.bitwise_or(
                mask_prods,
                mask_bbox_en_roi(img_h, img_w, x1, y1, x2, y2, mask_roi))

    area_huecos = int(cv2.countNonZero(mask_huecos))
    area_prods = int(cv2.countNonZero(mask_prods))

    # Unión (lo cubierto por PRODUCTOS o HUECOS, sin doble contar superposiciones)
    mask_union = cv2.bitwise_or(mask_prods, mask_huecos)
    area_union = int(cv2.countNonZero(mask_union))

    if modo == "huecos":
        pct_vacio = area_huecos / area_roi
        pct_ocupado = 1.0 - pct_vacio
    elif modo == "mixto":
        a = area_huecos
        b = max(0, area_roi - area_union)
        pct_vacio = max(a, b) / area_roi
        pct_ocupado = 1.0 - pct_vacio
    else:  # "productos"
        pct_ocupado = area_prods / area_roi
        pct_vacio = 1.0 - pct_ocupado

    pct_ocupado = float(np.clip(pct_ocupado, 0.0, 1.0))
    pct_vacio = float(np.clip(pct_vacio, 0.0, 1.0))
    return pct_ocupado, pct_vacio, area_roi, dets_validadas


def dibujar_panel(img, lineas, y0=20):
    """Overlay de texto con fondo semi-transparente arriba a la izquierda."""
    h_img = img.shape[0]
    max_w = max([cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                 for l in (lineas or [" "])], default=200)
    panel_w = min(720, max_w + 30)
    panel_h = y0 + 14 + len(lineas) * 26
    panel_h = min(panel_h, h_img - 10)

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, panel_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    for i, l in enumerate(lineas):
        y = y0 + i * 26
        if y >= panel_h - 6:
            break
        color = (255, 255, 255)
        if "ALERTA stock" in l or "REACTIVA" in l:
            color = (0, 80, 255)
        if "PRONÓSTICO" in l or "PROACTIVA" in l or "se vacía" in l or "se acaba" in l:
            color = (0, 165, 255)
        if "SLA(" in l and "OVER" in l:
            color = (0, 200, 255)
        cv2.putText(img, l, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)


def redimensionar(img, max_w=1280, max_h=760):
    h, w = img.shape[:2]
    escala = min(max_w / w, max_h / h, 1.0)
    if escala < 1.0:
        return cv2.resize(img, (int(w * escala), int(h * escala)))
    return img


def lista_imagenes(folder: Path):
    return sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])


def nothing(_):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="Path al modelo YOLO .pt entrenado (ej: models/runs/.../best.pt)")
    ap.add_argument("--image", help="Una sola imagen")
    ap.add_argument("--folder", help="Carpeta de imágenes (navegables con trackbar)")
    ap.add_argument("--video",
                    help="Archivo de video O 0/1 para cámara web (entero)")
    ap.add_argument("--camera", default="camara_01",
                    help="ID de cámara en configs/cameras.yaml (define los ROIs a usar)")
    ap.add_argument("--config", default=str(BASE / "configs" / "cameras.yaml"))
    ap.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza YOLO")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU NMS")
    ap.add_argument("--modo", choices=["productos", "huecos", "mixto"], default=None,
                    help="Override del cálculo de stock (por defecto lo lee de configs/cameras.yaml: stock.modo)")
    args = ap.parse_args()

    from ultralytics import YOLO  # import lazy
    modelo = YOLO(args.weights)

    cfg = cargar_yaml(args.config) or {}
    alarma_cfg = cfg.get("alarma", {}) or {}
    pct_vacio_min = float(alarma_cfg.get("porcentaje_vacio_minimo", 0.35))
    tiempo_sost = float(alarma_cfg.get("tiempo_sostenido_seg", 3.0))
    lat_max = float(alarma_cfg.get("latencia_maxima_seg", 5.0))

    enhancer = enhancer_desde_config(cfg)
    alerts = AlertManager(log_path=str(BASE / "data" / "alerts.log"))
    forecast = ForecastService()

    pronostico_cfg = cfg.get("pronostico", {}) or {}
    anticipada_min = float(pronostico_cfg.get("alertar_anticipada_si_t_menor_a_min", 30.0))
    forecast_h = int(pronostico_cfg.get("horizonte_minutos", forecast.horizonte_default))

    stock_cfg = cfg.get("stock", {}) or {}
    modo_stock = (args.modo or stock_cfg.get("modo") or "productos").lower()
    area_min_rel_hueco = float(stock_cfg.get("area_min_rel_hueco", 0.01))
    area_min_rel_producto = float(stock_cfg.get("area_min_rel_producto", 0.005))

    rois = []
    try:
        roi_handler = ROIHandler(config_path=args.config)
        rois = roi_handler.get_rois(args.camera)
        print(f"Usando ROIs de {args.camera}: {[r.id for r in rois]}")
    except Exception as e:
        print(f"[!] No se cargaron ROIs ({e}). Continuando sin ROIs.")

    # Preparar fuente de frames
    modo = None
    captura = None
    imgs_folder = []
    if args.image:
        modo = "image"
    elif args.folder:
        modo = "folder"
        imgs_folder = lista_imagenes(Path(args.folder))
        if not imgs_folder:
            print("No hay imágenes en", args.folder)
            return
        print(f"Carpeta: {len(imgs_folder)} imágenes")
    elif args.video is not None:
        modo = "video"
        src = args.video
        if src.isdigit():
            src = int(src)
        captura = cv2.VideoCapture(src)
    else:
        ap.error("Debés pasar --image, --folder o --video")

    # Estado de alerta sostenida
    estado_alerta = {
        roi.id: {
            "alerta_activa": False, "desde": None,
            "alertada_en_esta_ocasion": False,
            "anticipada_activa": False,
            "anticipada_desde": None,
            "anticipada_alertada": False,
        }
        for roi in rois
    }

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 760)
    if modo == "folder":
        cv2.createTrackbar("Img", WIN, 0, max(0, len(imgs_folder) - 1), nothing)

    ultima_clasif_roi: dict = {}
    ultima_forecast_roi: dict[str, ForecastResult] = {}
    ultima_inf_ms = 0.0
    ultima_pre_ms = 0.0

    try:
        while True:
            frame = None
            if modo == "image":
                frame = cv2.imread(args.image)
            elif modo == "folder":
                idx = cv2.getTrackbarPos("Img", WIN)
                idx = min(idx, len(imgs_folder) - 1)
                frame = cv2.imread(str(imgs_folder[idx]))
            elif modo == "video":
                ok, fr = captura.read()
                if not ok:
                    print("Fin del video / no se pudo leer cámara.")
                    break
                frame = fr

            if frame is None:
                print("No se pudo leer el frame. Saliendo.")
                break

            h, w = frame.shape[:2]

            # --- PIPELINE: preproc + inferencia ---
            t_pre0 = time.perf_counter()
            frame_proc = enhancer.process(frame)
            dt_pre = time.perf_counter() - t_pre0
            ultima_pre_ms = dt_pre * 1000.0

            t_inf0 = time.perf_counter()
            resultados = modelo.predict(frame_proc, conf=args.conf, iou=args.iou,
                                        verbose=False)
            dt_inf = time.perf_counter() - t_inf0
            ultima_inf_ms = dt_inf * 1000.0
            latencia_total_s = dt_pre + dt_inf

            dets = []
            if len(resultados) and resultados[0].boxes is not None:
                b = resultados[0].boxes
                cls_ids = b.cls.cpu().numpy().astype(int)
                confs = b.conf.cpu().numpy().astype(float)
                xyxy = b.xyxy.cpu().numpy()
                for i in range(len(cls_ids)):
                    x1, y1, x2, y2 = xyxy[i]
                    dets.append((float(x1), float(y1), float(x2), float(y2),
                                 int(cls_ids[i]), float(confs[i])))

            vis = frame_proc.copy()
            # (1) Primero dibujamos TODOS los ROIs (pasamos a pintar las detecciones DESPUÉS)
            #     Los colores definitivos los ponemos más abajo en el loop por ROI.
            # (2) Detecciones crudas -> las dibujamos sobre vis

            lineas_panel = []
            sla_ok = "OK" if latencia_total_s <= lat_max else "OVER"
            lineas_panel.append(
                f"cámara: {args.camera}  |  modo stock: {modo_stock}  |  preproc: {ultima_pre_ms:.0f} ms  "
                f"|  inferencia: {ultima_inf_ms:.0f} ms  "
                f"|  total E2E: {latencia_total_s*1000:.0f} ms  "
                f"SLA(<{lat_max*1000:.0f}ms): {sla_ok}"
            )
            lineas_panel.append(
                f"Forecast: horizonte {forecast_h} min  |  "
                f"Anticipada: si < {anticipada_min:.0f} min  |  "
                f"Histórico cada 60s en data/stock_history.csv"
            )
            lineas_panel.append(
                f"clases: {CLASSES}  |  umbral conf: {args.conf:.2f}  |  "
                f"área min: hueco={area_min_rel_hueco*100:.1f}% prod={area_min_rel_producto*100:.1f}% ROI"
            )
            lineas_panel.append("")

            ahora = time.time()
            # Diccionario de detecciones a dibujar por ROI (así solo pintamos las que están DENTRO de cada ROI)
            # y con el color_segun_tamano ya calculado.
            for roi in rois:
                pts = roi.a_pixeles(w, h)
                # --- Pre-computar mask_roi para reutilizarla en calcular_ocupacion ---
                mask_roi = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_roi, [pts], 255)
                area_roi = int(cv2.countNonZero(mask_roi))

                pct_ocu, pct_vac, area_roi, dets_validadas = calcular_ocupacion(
                    h, w, dets, pts,
                    modo=modo_stock,
                    area_min_rel_hueco=area_min_rel_hueco,
                    area_min_rel_producto=area_min_rel_producto,
                    area_roi=area_roi, mask_roi=mask_roi)

                # --- Dibujar detecciones (SOLO las de este ROI) con el color por tamaño ---
                for (x1, y1, x2, y2, cls, conf, area_rel, area_px, color) in dets_validadas:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    etiqueta = f"{CLASSES.get(cls, (str(cls), color))[0]} {conf:.2f} {area_rel*100:.0f}%"
                    (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (int(x1), int(y1) - th - 6),
                                  (int(x1) + tw + 6, int(y1)), color, -1)
                    cv2.putText(vis, etiqueta, (int(x1) + 3, int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                ultima_clasif_roi[roi.id] = (pct_ocu, pct_vac)
                alerta_ahora = pct_vac >= pct_vacio_min

                # --- LOGGING cada 1min al CSV de histórico (para forecast) ---
                forecast.loggear_muestra(MuestraStock(
                    camera_id=args.camera,
                    roi_id=roi.id,
                    pct_ocupado=pct_ocu,
                    pct_vacio=pct_vac,
                    num_detecciones=len(dets),
                    latencia_e2e_s=latencia_total_s,
                ))

                f_res = forecast.predecir_ahora(
                    cam_id=args.camera,
                    roi_id=roi.id,
                    pct_ocupado_actual=pct_ocu,
                    horizonte_min=forecast_h,
                    recalcular_si_hace_falta=(modo == "video"),
                )
                ultima_forecast_roi[roi.id] = f_res
                t_min = f_res.tiempo_hasta_vacio_min
                pronosticada_ahora = (t_min is not None and t_min <= anticipada_min)

                est = estado_alerta[roi.id]
                if modo == "video":
                    if alerta_ahora:
                        if est["desde"] is None:
                            est["desde"] = ahora
                        llevamos = ahora - est["desde"]
                        est["alerta_activa"] = llevamos >= tiempo_sost
                    else:
                        est["desde"] = None
                        est["alerta_activa"] = False
                        est["alertada_en_esta_ocasion"] = False
                else:
                    est["alerta_activa"] = alerta_ahora
                    est["alertada_en_esta_ocasion"] = False

                if modo == "video":
                    if pronosticada_ahora:
                        if est["anticipada_desde"] is None:
                            est["anticipada_desde"] = ahora
                        llevamos_a = ahora - est["anticipada_desde"]
                        est["anticipada_activa"] = llevamos_a >= 10.0
                    else:
                        est["anticipada_desde"] = None
                        est["anticipada_activa"] = False
                        est["anticipada_alertada"] = False
                else:
                    est["anticipada_activa"] = pronosticada_ahora
                    est["anticipada_alertada"] = False

                md = {
                    "pct_ocupado": round(pct_ocu, 4),
                    "area_roi_px": area_roi,
                    "num_detecciones": len(dets),
                    "latencia_e2e_s": round(latencia_total_s, 4),
                    "modo": modo,
                }
                if t_min is not None:
                    md["forecast_min_hasta_vacio"] = round(t_min, 2)
                    md["forecast_metodo"] = f_res.metodo_usado
                    md["forecast_conf"] = round(f_res.nivel_confianza, 3)
                    if f_res.fecha_estimada_vacio is not None:
                        md["forecast_fecha_local"] = f_res.fecha_estimada_vacio.strftime("%Y-%m-%d %H:%M:%S")

                # Alerta REACTIVA
                if est["alerta_activa"] and not est.get("alertada_en_esta_ocasion", False):
                    md_r = dict(md); md_r["tipo"] = "REACTIVA_umbral_vacio"
                    evento = alerts.alertar(
                        camera_id=args.camera, roi_id=roi.id,
                        pct_vacio=pct_vac, pct_umbral=pct_vacio_min,
                        metadatos=md_r,
                    )
                    if evento is not None:
                        est["alertada_en_esta_ocasion"] = True

                # Alerta PROACTIVA
                if est["anticipada_activa"] and not est.get("anticipada_alertada", False):
                    md_a = dict(md); md_a["tipo"] = "PROACTIVA_forecast"
                    # pct_umbral para la proactiva lo usamos como "marcador" distinguible
                    # en el CSV (no se usa como criterio porque el criterio ya es el tiempo).
                    pct_umbral_proactiva = max(0.01, pct_vacio_min * 0.2)  # solo distintivo
                    evento_a = alerts.alertar(
                        camera_id=args.camera, roi_id=roi.id,
                        pct_vacio=pct_vac, pct_umbral=pct_umbral_proactiva,
                        metadatos=md_a,
                    )
                    if evento_a is not None:
                        est["anticipada_alertada"] = True

                if est["alerta_activa"]:
                    color_roi = (0, 0, 255)
                    espesor = 4
                elif est["anticipada_activa"]:
                    color_roi = (0, 165, 255)
                    espesor = 3
                else:
                    color_roi = (255, 180, 0)
                    espesor = 2
                cv2.polylines(vis, [pts], isClosed=True, color=color_roi, thickness=espesor)
                cx = float(pts[:, 0].mean())
                cy = int(pts[:, 1].min()) - 8
                cv2.putText(vis, roi.id, (int(cx) - 40, max(20, int(cy))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_roi, 2, cv2.LINE_AA)

                estado_texto = "[ALERTA stock]" if est["alerta_activa"] else \
                    ("[PRONÓSTICO: se acaba]" if est["anticipada_activa"] else "OK")
                if modo == "video" and est["desde"] is not None and not est["alerta_activa"]:
                    restante = max(0.0, tiempo_sost - (ahora - est["desde"]))
                    estado_texto += f" (contando {restante:.1f}s)"
                if modo == "video" and (not est["anticipada_activa"]) and est["anticipada_desde"] is not None:
                    ra = max(0.0, 10.0 - (ahora - est["anticipada_desde"]))
                    estado_texto += f" (Pron en {ra:.1f}s)"

                t_min = f_res.tiempo_hasta_vacio_min
                eta_str = "-"
                if f_res.fecha_estimada_vacio is not None:
                    eta_str = f_res.fecha_estimada_vacio.strftime("%H:%M")
                if t_min is None:
                    pronostico_txt = (
                        f"{f_res.mensaje_humano}  [método: {f_res.metodo_usado}  "
                        f"ETA {eta_str}  conf={f_res.nivel_confianza:.2f}]"
                    )
                else:
                    pronostico_txt = (
                        f"{f_res.mensaje_humano}  [método: {f_res.metodo_usado}  "
                        f"ETA {eta_str} (t={t_min:.0f}min)  conf={f_res.nivel_confianza:.2f}]"
                    )

                lineas_panel.append(
                    f"ROI {roi.id}: Ocupado={pct_ocu*100:5.1f}%  "
                    f"Vacio={pct_vac*100:5.1f}%  (umbral vacío={pct_vacio_min*100:.0f}%) {estado_texto}"
                )
                lineas_panel.append(f"       ↳ {pronostico_txt}")

            dibujar_panel(vis, lineas_panel, y0=30)
            vis = redimensionar(vis)

            if modo in ("folder", "image"):
                footer = f"q=salir  |  s=guardar captura  |  {len(dets)} detecciones  |  preproc+forecast ON"
            else:
                fps_pipe = 1000.0 / max(1.0, (ultima_pre_ms + ultima_inf_ms))
                footer = f"q=salir  |  s=guardar captura  |  FPS pipeline: ~{fps_pipe:.0f}"
            cv2.putText(vis, footer, (20, vis.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(WIN, vis)
            k = cv2.waitKey(1 if modo == "video" else 50) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                out = BASE / "docs" / f"demo_{int(time.time())}.png"
                out.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(out), vis)
                print(f"Captura guardada: {out}")

            if modo == "image":
                while (cv2.waitKey(100) & 0xFF) not in (ord("q"),):
                    pass
                break
    finally:
        alerts.close()

    if captura is not None:
        captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
