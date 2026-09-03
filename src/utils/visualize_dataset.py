"""
Ventana interactiva para visualizar el dataset YOLO (ground-truth).
Útil para verificar que las anotaciones (labels .txt) coinciden con las imágenes.

Controles:
  - Trackbar "Split": cambia entre TRAIN (0) y VAL (1)
  - Trackbar "Img": navega por las imágenes del split actual
  - Tecla "q": salir
  - Tecla "s": guardar captura en docs/<split>_<idx>.png

Uso:
  python src/utils/visualize_dataset.py
  python src/utils/visualize_dataset.py --base data/processed
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent

CLASSES = {0: ("vacio", (0, 0, 255)),    # rojo
           1: ("producto", (0, 200, 0))}  # verde

WIN = "Dataset YOLO - Ground Truth"


def listar_split(base: Path, split: str):
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split
    imgs = sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    pares = []
    for imp in imgs:
        lbl = lbl_dir / (imp.stem + ".txt")
        pares.append((imp, lbl if lbl.exists() else None))
    return pares


def leer_labels_yolo(path_lbl: Path | None, img_w: int, img_h: int):
    bboxes = []
    if path_lbl is None or not path_lbl.exists():
        return bboxes
    with open(path_lbl, "r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.strip().split()
            if len(partes) < 5:
                continue
            cls = int(partes[0])
            cx, cy, w, h = map(float, partes[1:5])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            bboxes.append((cls, x1, y1, x2, y2))
    return bboxes


def dibujar(img, bboxes):
    out = img.copy()
    for cls, x1, y1, x2, y2 in bboxes:
        nombre, color = CLASSES.get(cls, (f"cls{cls}", (255, 255, 0)))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, nombre, (x1, max(y1 - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def redimensionar(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    escala = min(max_w / w, max_h / h, 1.0)
    if escala < 1.0:
        nw, nh = int(w * escala), int(h * escala)
        return cv2.resize(img, (nw, nh))
    return img


def nothing(_):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE / "data" / "processed"))
    args = ap.parse_args()
    base = Path(args.base).resolve()

    pares_split = {
        "train": listar_split(base, "train"),
        "val": listar_split(base, "val"),
    }

    print(f"TRAIN: {len(pares_split['train'])} imágenes")
    print(f"VAL  : {len(pares_split['val'])} imágenes")
    if sum(len(v) for v in pares_split.values()) == 0:
        print("ERROR: no hay imágenes. Revisá --base")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.createTrackbar("Split [0=train 1=val]", WIN, 0, 1, nothing)
    cv2.createTrackbar("Img", WIN, 0, max(1, len(pares_split["train"]) - 1), nothing)

    ultimo_split = -1
    while True:
        split_idx = cv2.getTrackbarPos("Split [0=train 1=val]", WIN)
        split_nombre = "train" if split_idx == 0 else "val"
        lista = pares_split[split_nombre]
        if not lista:
            cv2.putText(np.zeros((400, 800, 3), dtype=np.uint8),
                        f"Split '{split_nombre}' vacío", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            continue

        if split_nombre != ultimo_split:
            max_idx = max(0, len(lista) - 1)
            # Trackbar máximo = máximo índice
            try:
                cv2.setTrackbarMax("Img", WIN, max_idx)
                cv2.setTrackbarPos("Img", WIN, 0)
            except Exception:
                pass
            ultimo_split = split_nombre

        idx = cv2.getTrackbarPos("Img", WIN)
        idx = min(idx, len(lista) - 1)

        impath, lblpath = lista[idx]
        img = cv2.imread(str(impath))
        if img is None:
            print(f"No se pudo leer: {impath}")
            k = cv2.waitKey(50) & 0xFF
            if k == ord("q"):
                break
            continue

        h, w = img.shape[:2]
        bboxes = leer_labels_yolo(lblpath, w, h)
        vis = dibujar(img, bboxes)
        vis = redimensionar(vis)

        # Barra superior con info
        info = f"{split_nombre.upper()} [{idx + 1}/{len(lista)}]  {impath.name}  " \
               f"boxes={len(bboxes)}  WxH={w}x{h}"
        cv2.putText(vis, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (20, 20, 200), 1, cv2.LINE_AA)
        ayuda = "q=salir | s=guardar captura | trackbars: cambiar split / imagen"
        cv2.putText(vis, ayuda, (12, vis.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow(WIN, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            out = BASE / "docs" / f"{split_nombre}_{idx:03d}.png"
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), vis)
            print(f"Guardada captura: {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
