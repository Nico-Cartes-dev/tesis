"""
Pseudo-etiquetado automático de HUECOS VACÍOS (clase 0 = vacio) a partir de
labels ya existentes de PRODUCTOS (clase 1 = producto) en formato YOLO.

Funcionamiento (por cada imagen):
  1. Lee la imagen → W, H.
  2. Lee su label .txt YOLO: lineas = [cls cx cy w h, ...] (todo normalizado 0-1).
  3. Rellenar de '1' (producto) todo lo cubierto por bounding boxes de clase 1.
  4. Rellenar de '0' (hueco) TODO LO DEMÁS (lo que no es producto).
  5. Encontrar componentes conexas (contornos) del '0' → aproximar cada una a un
     rectángulo axis-aligned (bounding box externo).
  6. Escribir la NUEVA label .txt (o sobreescribir --sobrescribir) manteniendo las
     lineas originales de clase 1 y sumando las lineas de clase 0 (huecos).

Opcional:
  --solo-huecos  : en el .txt de salida SÓLO guardar clase 0 (no guardar clase 1)
  --sobrescribir : reemplazar el .txt original (por defecto guarda backup .bak)

Uso:
  python src/utils/pseudo_label_huecos.py --base data/processed
  python src/utils/pseudo_label_huecos.py --base data/processed --sobrescribir --solo-huecos
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp")


def leer_yolo_txt(p: Path):
    lines = []
    if not p.exists():
        return lines
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            row = raw.strip().split()
            if len(row) < 5:
                continue
            try:
                cls = int(row[0])
                cx, cy, w, h = [float(x) for x in row[1:5]]
                lines.append((cls, cx, cy, w, h, raw.strip()))
            except Exception:
                continue
    return lines


def yolo_xyxy(cx, cy, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    bx = cx * img_w
    by = cy * img_h
    x1 = max(0, int(bx - bw / 2.0))
    y1 = max(0, int(by - bh / 2.0))
    x2 = min(img_w, int(bx + bw / 2.0))
    y2 = min(img_h, int(by + bh / 2.0))
    return x1, y1, x2, y2


def bbox_a_yolo(x1, y1, x2, y2, img_w, img_h):
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    cx = float(x1) + bw / 2.0
    cy = float(y1) + bh / 2.0
    return (
        round(cx / img_w, 6),
        round(cy / img_h, 6),
        round(bw / img_w, 6),
        round(bh / img_h, 6),
    )


def procesar_imagen(img_path: Path, lbl_path: Path, out_lbl_path: Path,
                    solo_huecos: bool, area_min_rel: float = 0.005,
                    margen_px: int = 2) -> tuple[int, int]:
    """
    Devuelve (# productos originales conservados, # huecos nuevos agregados).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0
    H, W = img.shape[:2]

    labels = leer_yolo_txt(lbl_path)
    mask_productos = np.zeros((H, W), dtype=np.uint8)

    for cls, cx, cy, w, h, _raw in labels:
        if cls == 1:
            x1, y1, x2, y2 = yolo_xyxy(cx, cy, w, h, W, H)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(mask_productos, (x1, y1), (x2, y2), 255, -1)

    # Huecos = todo lo que NO es producto (dentro de la imagen)
    mask_huecos = cv2.bitwise_not(mask_productos)

    # (Opcional) quitamos un margen de 1-2px alrededor de la imagen para no
    # considerar "hueco" los bordes fuera de la góndola.
    if margen_px > 0:
        mask_borde = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask_borde, (margen_px, margen_px),
                      (W - margen_px, H - margen_px), 255, -1)
        mask_huecos = cv2.bitwise_and(mask_huecos, mask_borde)

    # Componentes conexas de los huecos (8-conectividad)
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_huecos, connectivity=8)

    area_min = max(1, int(area_min_rel * H * W))
    huecos_xyxy = []
    for i in range(1, n_labels):  # skip fondo (label 0)
        x, y, bw, bh, area = stats[i]
        if area < area_min:
            continue
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W, int(x + bw))
        y2 = min(H, int(y + bh))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        huecos_xyxy.append((x1, y1, x2, y2))

    # Preparar lineas del nuevo label
    nuevas_lineas: list[str] = []
    n_prod = 0
    if not solo_huecos:
        for cls, cx, cy, w, h, raw in labels:
            if cls == 1:
                nuevas_lineas.append(raw)
                n_prod += 1
            # Otras clases (distintas de 1) se preservan a menos que sea clase 0
            elif cls != 0:
                nuevas_lineas.append(raw)

    # Agregamos clase 0 (huecos)
    n_huecos = 0
    for (x1, y1, x2, y2) in huecos_xyxy:
        cx_n, cy_n, w_n, h_n = bbox_a_yolo(x1, y1, x2, y2, W, H)
        if w_n <= 0.0 or h_n <= 0.0:
            continue
        nuevas_lineas.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
        n_huecos += 1

    # Backup previo si no se pide sobrescribir y el archivo existía
    if lbl_path.exists():
        if not (out_lbl_path.parent / (out_lbl_path.name + ".bak")).exists():
            shutil.copy2(lbl_path, lbl_path.parent / (lbl_path.name + ".bak"))

    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(nuevas_lineas))
        if nuevas_lineas:
            f.write("\n")

    return n_prod, n_huecos


def procesar_split(split_dir_img: Path, split_dir_lbl: Path, **kwargs) -> tuple[int, int, int]:
    if not split_dir_img.exists():
        return 0, 0, 0
    imagenes = sorted([p for p in split_dir_img.iterdir()
                       if p.suffix.lower() in EXT_IMG])
    total_img = 0
    total_prod = 0
    total_huecos = 0
    for img_p in imagenes:
        lbl_p = split_dir_lbl / (img_p.stem + ".txt")
        out_p = lbl_p  # escribimos en la misma ruta (con backup previo dentro)
        np_, nh_ = procesar_imagen(img_p, lbl_p, out_p, **kwargs)
        total_img += 1
        total_prod += np_
        total_huecos += nh_
        print(f"  [{split_dir_img.parent.name}/{split_dir_img.name}] {img_p.name}  "
              f"→ productos={np_}, huecos_nuevos={nh_}")
    return total_img, total_prod, total_huecos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/processed",
                    help="Directorio raíz con images/ y labels/ (train + val)")
    ap.add_argument("--area-min-rel", type=float, default=0.005,
                    help="Ignorar huecos con area menor a este ratio (default 0.005 = 0.5 por ciento de la imagen)")
    ap.add_argument("--margen-px", type=int, default=2,
                    help="Ignorar borde de N pixeles alrededor de la imagen (evita bordes como huecos)")
    ap.add_argument("--solo-huecos", action="store_true",
                    help="Solo guardar clase 0 (borra las detecciones de producto del .txt)")
    args = ap.parse_args()

    base = Path(args.base)
    resumen = {}
    for split in ("train", "val"):
        img_d = base / "images" / split
        lbl_d = base / "labels" / split
        if not img_d.exists():
            print(f"[skip] {img_d} no existe")
            continue
        print(f"=== Split: {split} ===")
        ti, tp, th = procesar_split(img_d, lbl_d,
                                    solo_huecos=args.solo_huecos,
                                    area_min_rel=args.area_min_rel,
                                    margen_px=args.margen_px)
        resumen[split] = (ti, tp, th)

    print("\n=== RESUMEN ===")
    for s, (ti, tp, th) in resumen.items():
        print(f"  {s}: {ti} imágenes  |  productos conservados={tp}  |  huecos nuevos={th}")
    print("Hecho. Los .txt originales tienen backup en: <nombre>.txt.bak")


if __name__ == "__main__":
    main()
