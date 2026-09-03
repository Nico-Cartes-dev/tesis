"""
Convierte los datasets crudos (Supermarket shelves + dataset 2) a formato YOLO,
hace split train/val y coloca todo en data/processed/.

Origen:
  datos de entrenamiento/Supermarket shelves/  (imágenes + .json Supervisely)
  datos de entrenamiento/dataset 2/            (imágenes + annotations.xml CVAT)

Destino:
  data/processed/images/train/
  data/processed/images/val/
  data/processed/labels/train/
  data/processed/labels/val/

Clases YOLO (según configs/dataset.yaml):
  0 = vacio
  1 = producto
"""

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
SRC = BASE / "datos de entrenamiento"
DEST = BASE / "data" / "processed"

RANDOM_SEED = 42
TRAIN_RATIO = 0.80


def xyxy_a_yolo(x1: float, y1: float, x2: float, y2: float,
                img_w: int, img_h: int):
    """Convierte bbox de esquinas (píxeles) a YOLO normalizado [cx, cy, w, h]."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return [cx / img_w, cy / img_h, bw / img_w, bh / img_h]


def procesar_supermarket_shelves():
    """Lee el dataset 'Supermarket shelves' (JSON Supervisely)."""
    muestras = []
    img_dir = SRC / "Supermarket shelves" / "images"
    ann_dir = SRC / "Supermarket shelves" / "annotations"

    if not img_dir.exists():
        print(f"[!] No existe: {img_dir}")
        return muestras

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        json_path = ann_dir / f"{img_path.name}.json"
        if not json_path.exists():
            print(f"[!] Falta anotación para {img_path.name}, salteando.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = int(data["size"]["width"])
        img_h = int(data["size"]["height"])

        bboxes = []
        for obj in data.get("objects", []):
            # classTitle = "Product" => clase 1 (producto)
            if obj.get("classTitle", "").lower() != "product":
                continue
            ext = obj["points"]["exterior"]
            x1, y1 = ext[0]
            x2, y2 = ext[1]
            yolo = xyxy_a_yolo(x1, y1, x2, y2, img_w, img_h)
            bboxes.append((1, *yolo))

        if bboxes:
            muestras.append((f"sm_{img_path.name}", img_path, bboxes))

    return muestras


def procesar_dataset2():
    """Lee 'dataset 2' (XML CVAT)."""
    muestras = []
    img_dir = SRC / "dataset 2" / "img"
    xml_path = SRC / "dataset 2" / "annotations.xml"

    if not xml_path.exists() or not img_dir.exists():
        print(f"[!] No existen {xml_path} o {img_dir}, salteando dataset2.")
        return muestras

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image_tag in root.findall("image"):
        # name en XML suele ser "image/X.png" → extraemos nombre final
        name_attr = image_tag.attrib["name"]
        base_name = Path(name_attr).name  # ej: "1.png"
        img_w = int(image_tag.attrib["width"])
        img_h = int(image_tag.attrib["height"])

        # Buscamos el archivo real en img_dir
        candidates = list(img_dir.glob(f"{Path(base_name).stem}.*"))
        if not candidates:
            print(f"[!] No se encuentra imagen para {base_name}, salteando.")
            continue
        img_path = candidates[0]

        bboxes = []
        for poly in image_tag.findall("polyline"):
            if poly.attrib.get("label", "") != "Box":
                continue
            puntos_str = poly.attrib["points"]  # "x1,y1;x2,y2;x3,y3;x4,y4"
            pts = []
            for p in puntos_str.split(";"):
                x, y = p.split(",")
                pts.append((float(x), float(y)))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            yolo = xyxy_a_yolo(x1, y1, x2, y2, img_w, img_h)
            bboxes.append((1, *yolo))  # "Box" => producto clase 1

        if bboxes:
            muestras.append((f"ds2_{base_name}", img_path, bboxes))

    return muestras


def escribir_label(path_label: Path, bboxes):
    path_label.parent.mkdir(parents=True, exist_ok=True)
    with open(path_label, "w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in bboxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    random.seed(RANDOM_SEED)

    print("-> Procesando Supermarket shelves...")
    sm = procesar_supermarket_shelves()
    print(f"   OK: {len(sm)} imágenes con productos.")

    print("-> Procesando dataset 2...")
    ds2 = procesar_dataset2()
    print(f"   OK: {len(ds2)} imágenes con productos.")

    todas = sm + ds2
    random.shuffle(todas)

    n = len(todas)
    cut = int(n * TRAIN_RATIO)
    train_set = todas[:cut]
    val_set = todas[cut:]
    print(f"\nTotal: {n} imágenes → train={len(train_set)}, val={len(val_set)}")

    # Limpiamos destinos antes
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        p = DEST / sub
        if p.exists():
            for f in p.iterdir():
                if f.is_file():
                    f.unlink()

    for split_name, dataset in [("train", train_set), ("val", val_set)]:
        img_dir_out = DEST / "images" / split_name
        lbl_dir_out = DEST / "labels" / split_name
        img_dir_out.mkdir(parents=True, exist_ok=True)
        lbl_dir_out.mkdir(parents=True, exist_ok=True)

        for nombre_dest, src_img, bboxes in dataset:
            # Copiamos imagen
            dest_img = img_dir_out / nombre_dest
            shutil.copy2(src_img, dest_img)
            # Escribimos label .txt
            dest_lbl = (lbl_dir_out / nombre_dest).with_suffix(".txt")
            escribir_label(dest_lbl, bboxes)

    print("\nListo. Dataset YOLO en:", DEST)


if __name__ == "__main__":
    main()
