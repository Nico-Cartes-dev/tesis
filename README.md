# Shelf Monitor — Detección de falta de stock en góndola

Proyecto de computer vision para detectar **zonas de góndola con stock faltante** usando:
- Cámaras de seguridad (baja resolución) + **preprocesamiento** (denoise + CLAHE + gamma opcional) basado en Gonzalez & Woods.
- **YOLOv8** entrenado con clase `producto` y **clase `vacio` (huecos en la línea)**.
- **Tres modos de cálculo** de `%vacío / %ocupado`:
  - `productos` → YOLO detecta productos (clásico)
  - `huecos` → YOLO detecta la línea vacía (tesis, objetivo principal)
  - `mixto` → combina ambos, más robusto
- **AlertManager**: alertas **reactivas** (umbral %vacío + tiempo sostenido) y **proactivas / predictivas**
  ("está por acabarse en X min").
- **Forecast**: rolling slope (desde el minuto 1) + Prophet/SARIMA fallback cuando haya ≥ 200
  muestras históricas. Ranking semanal de "repisas con mayor tendencia a agotarse".
- SLA de latencia end-to-end: ideal 1s / máximo 5s.

## Estructura

```
tesis/
├── data/
│   ├── raw/                          # imágenes/video crudo de cámaras (NO subir a Git)
│   ├── processed/
│   │   ├── images/
│   │   │   ├── train/                # imágenes train YOLO
│   │   │   └── val/                  # imágenes val YOLO
│   │   └── labels/
│   │       ├── train/                # etiquetas .txt YOLO train
│   │       └── val/                  # etiquetas .txt YOLO val
│   └── labels/                       # etiquetas intermedias / exportadas
├── src/
│   ├── preprocessing/
│   │   ├── image_enhancer.py         # denoise → CLAHE → gamma → sharpen
│   │   └── roi_handler.py            # ROIs rectángulo / polígono normalizados [0,1]
│   ├── detection/
│   │   ├── train.py                  # entrenamiento YOLO + seeds reproducibles
│   │   └── detect.py                 # demo interactiva ventana OpenCV (3 modos stock)
│   ├── alerts/
│   │   ├── alert_manager.py          # salida única de alertas (CSV/consola/callback)
│   │   └── llm_summary.py            # resumen en lenguaje natural (OFFLINE/ONLINE)
│   ├── prediction/
│   │   ├── stock_history.py          # CSV histórico de %ocupado cada N segundos
│   │   ├── forecast_models.py        # RollingSlope / Prophet / SARIMA / fallback
│   │   ├── forecast_service.py       # orquestación + persistencia forecasters .pkl
│   │   └── top_riesgo.py             # ranking TOP 5 riesgo semanal
│   └── utils/
│       ├── convert_dataset.py        # conversión Supervisely/CVAT → YOLO
│       ├── visualize_dataset.py      # ventana con trackbars para ver GT del dataset
│       ├── pseudo_label_huecos.py    # pseudo-etiqueta CLASE 0 (huecos) desde labels de producto
│       └── seed_demo_forecast.py     # seed sintético de 1h para probar forecast sin cámara real
├── configs/
│   ├── cameras.yaml                  # ROIs por cámara + preproc + stock.modo + umbrales
│   └── dataset.yaml                  # clases YOLO y paths train/val
├── models/                           # pesos entrenados (NO subir a Git)
│   └── forecasters/                  # forecasters persistidos (RollingSlope/Prophet/SARIMA .pkl)
├── notebooks/
├── tests/                            # 21 tests unitarios (ROI, ImageEnhancer, Alerts, Forecast)
├── docs/
│   ├── flow.md                       # diagrama de flujo Mermaid completo
│   └── ranking_riesgo_semanal.md     # ejemplo de ranking TOP 5 (generable)
├── kickoff-notes.md
├── requirements.txt
└── .gitignore
```

---

## 1. Setup inicial

```powershell
# Crea entorno virtual (Windows, PowerShell)
cd C:\Users\nicolas\Desktop\tesis
& "C:\Python311\python.exe" -m venv .venv

# Actívalo (o usá siempre la ruta .venv\Scripts\python.exe como hacemos aquí)
& ".\.venv\Scripts\Activate.ps1"

# Instala dependencias
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

> **Nota**: si `Activate.ps1` se bloquea por política de ejecución, no hace falta activar el entorno:
> usá siempre `& "C:\Users\nicolas\Desktop\tesis\.venv\Scripts\python.exe"` delante de los comandos.

---

## 2. Preparar el dataset YOLO (desde 0 o desde etiquetas de producto ya existentes)

### 2.1 Empezar desde imágenes crudas sin etiquetar
1. Colocá las fotos crudas en `data/raw/`.
2. Etiquetá con [LabelImg](https://github.com/heartexlabs/labelImg), [CVAT](https://www.cvat.ai/) o [Roboflow](https://roboflow.com/) — exportá en formato YOLO.
   - **Clase 0 → `vacio` (hueco en la línea, objetivo principal de la tesis)**
   - **Clase 1 → `producto` (opcional para modo `mixto`)**
3. Organizalo en `data/processed/images/{train,val}` y `data/processed/labels/{train,val}`.

### 2.2 Reutilizar etiquetas antiguas SÓLO de productos (generar huecos automáticamente)
Si ya tenés etiquetada **solo la clase `1 = producto`**, no hace falta re-etiquetar a mano:
```powershell
& ".\.venv\Scripts\python.exe" src/utils/pseudo_label_huecos.py --base data/processed --area-min-rel 0.008 --margen-px 6
```
Esto:
- Genera **clase 0 (huecos)** automáticamente como "todo lo que NO es producto en la imagen".
- Mantiene tus clases 1 intactas, así que quedan las DOS clases en el mismo `.txt` (ideal para `stock.modo = 'mixto'`).
- Crea backup de cada `.txt` original como `<archivo>.txt.bak` por si querés volver atrás.

### 2.3 Ver visualmente que las etiquetas hayan quedado bien
```powershell
& ".\.venv\Scripts\python.exe" src/utils/visualize_dataset.py
```
- Trackbar **Split**: `0 = train`, `1 = val`.
- Trackbar **Img**: navegá por las imágenes.
- `q` = salir, `s` = guardar captura en `docs/`.

---

## 3. Configurar ROIs y parámetros

Editá `configs/cameras.yaml` y definí, por cada `cámara`:
- `id`: nombre interno de cámara.
- `rois[]`: zonas de la repisa (rectángulo o polígono, **coordenadas normalizadas 0-1** de resolución).
- `preprocesamiento`: activar/ajustar denoise / CLAHE / gamma / sharpen.
- `stock.modo`: `'huecos'` (tesis) / `'productos'` / `'mixto'` (más robusto).
- `stock.area_min_rel_hueco` y `stock.area_min_rel_producto`: umbrales bajo los cuales los bboxes se
  descartan como chamusca / ruido.
- `alarma`: `porcentaje_vacio_minimo`, `tiempo_sostenido_seg`, `latencia_maxima_seg`.
- `pronostico`: `horizonte_minutos`, `alertar_anticipada_si_t_menor_a_min`, `muestreo_historico_cada_seg`.

Verificá los ROIs dibujados sobre una imagen real:
```powershell
& ".\.venv\Scripts\python.exe" src/preprocessing/roi_handler.py camara_01 data/raw/tu_foto.jpg
```
Sale `debug_rois.jpg` en el dir actual.

---

## 4. Probar el preprocesamiento en una imagen

```powershell
& ".\.venv\Scripts\python.exe" src/preprocessing/image_enhancer.py data/raw/tu_foto.jpg
```
Crea `*_processed.jpg`. Compará contra la original y ajustá:
- `preprocesamiento.denoise.fuerza`
- `preprocesamiento.clahe.clip_limit` y `tile_grid_size`
- `preprocesamiento.gamma.activo=True` y `gamma_valor` (por ejemplo `0.8` o `1.2`) si el sensor de la cámara tiene curva gamma rara.

---

## 5. Entrenar YOLO (con dataset de huecos + opcionalmente productos)

```powershell
& ".\.venv\Scripts\python.exe" src/detection/train.py --data configs/dataset.yaml --epochs 80 --batch 8 --nombre stock_huecos_v1
```
- Usa `yolov8n.pt` pre-entrenado por defecto.
- Seeds fijados (`random/numpy/torch` + `deterministic=True`) para resultados **reproducibles**.
- `patience=20` (early stopping si no mejora validación en 20 epochs).
- El mejor modelo queda en:
  ```
  models/runs/stock_huecos_v1/weights/best.pt
  ```
  (y también `last.pt` por si querés resumir desde el último checkpoint).

---

## 6. Demo interactiva (ventana OpenCV) — 4 formas de usarla

El **mismo script** soporta 4 fuentes y 3 modos de cálculo de stock.

### 6a) Sobre carpeta de validación (lo más usado para probar primero)
```powershell
& ".\.venv\Scripts\python.exe" src/detection/detect.py `
    --weights models/runs/stock_huecos_v1/weights/best.pt `
    --folder data/processed/images/val `
    --camera camara_02
```
- Trackbar **Img**: navegá por las 12 imágenes de val.
- Cada caja de **hueco** se colorea según su % del área del ROI:
  - `<5% verde` · `5-20% amarillo` · `20-35% naranja` · `>35% rojo`
- Cada **ROI** se dibuja:
  - Azul cian → OK
  - NARANJA → Alerta proactiva (forecast: se acaba en < 30 min)
  - ROJO → Alerta reactiva (%vacío > umbral y sostenido ≥ 3s)
- Panel superior: `preproc ms / inferencia ms / E2E ms + SLA` + `Forecast horizonte` +
  `modo stock: mixto/huecos/productos`.

### 6b) Forzar un modo concreto (override CLI)
```powershell
# SOLO cuenta huecos (tu idea original de "detectar la línea")
& ".\.venv\Scripts\python.exe" src/detection/detect.py `
    --weights models/runs/stock_huecos_v1/weights/best.pt `
    --folder data/processed/images/val `
    --camera camara_02 `
    --modo huecos

# Solo cuenta productos (backward-compatible)
& ".\.venv\Scripts\python.exe" src/detection/detect.py `
    --weights models/runs/stock_huecos_v1/weights/best.pt `
    --folder data/processed/images/val `
    --modo productos
```

### 6c) Una sola imagen + ROI de una cámara
```powershell
& ".\.venv\Scripts\python.exe" src/detection/detect.py `
    --weights models/runs/stock_huecos_v1/weights/best.pt `
    --image data/raw/tu_foto.jpg `
    --camera camara_01
```

### 6d) En VIVO (video / cámara web) — **activa el forecast real**
```powershell
# 0 = cámara default de tu laptop
& ".\.venv\Scripts\python.exe" src/detection/detect.py `
    --weights models/runs/stock_huecos_v1/weights/best.pt `
    --video 0 `
    --camera camara_02
```
- Cada **`muestreo_historico_cada_seg`** (default 60s = 1 min) guarda una fila en
  `data/stock_history.csv`.
- Cuando tengas ≥ 20 filas de ese ROI → empieza el forecast por **RollingSlope**.
- Cuando tengas ≥ 67 filas → fallback a **SARIMA** si tenés `statsmodels`.
- Cuando tengas ≥ 200 filas → usa **Prophet** si tenés `prophet` instalado (captura daily/weekly seasonality).

---

## 7. Alertas

Todas las alertas salen por **un único punto**: `src/alerts/alert_manager.py`.
- Se registran en **CSV** (`data/alerts.log`) y por **consola**.
- Tipos:
  - `REACTIVA_umbral_vacio` → %vacío > umbral + tiempo sostenido.
  - `PROACTIVA_forecast` → `tiempo estimado hasta vacío < alertar_anticipada_si_t_menor_a_min`.
- Tienen **cooldown por `(cámara, roi)`** (por defecto 60s) para no spamear.
- Se puede enchufar **cualquier canal externo** sin tocar `detect.py`, vía `AlertManager(extra_callback=...)`.
  Ejemplo de **canal Telegram**: `docs/flow.md` tiene un snippet listo para copiar.

---

## 8. Pronóstico + Ranking de riesgo (qué producto se acaba más rápido)

### 8.1 Ranking TOP 5 semanal
```powershell
& ".\.venv\Scripts\python.exe" src/prediction/top_riesgo.py
```
Muestra por consola las **5 repisas con mayor riesgo de ruptura de stock**, ordenadas por
`score = veces_alerta * 100 + |slope %/min| * 1000 - ocup_med %`.

Exportalo a markdown para la tesis / el encargado:
```powershell
& ".\.venv\Scripts\python.exe" src/prediction/top_riesgo.py --exportar-md docs/ranking_riesgo_semanal.md
```

### 8.2 Resumen LLM en lenguaje natural de una alerta
- **Modo OFFLINE (sin API key, siempre funciona)**:
  ```powershell
  & ".\.venv\Scripts\python.exe" src/alerts/llm_summary.py
  ```
  Imprime ejemplo de alerta proactiva + reactiva formateada para WhatsApp/email al encargado.
- **Modo ONLINE (OpenAI / OpenRouter-compatible)**: en `src/alerts/llm_summary.py` creá
  `ResumidorAlertasLLM(cliente=<tu_cliente_openai>)` y usá `resumir_alerta(...)`.
- Para usarlo **dentro del pipeline real** (y que cada alerta se envía resumida por Telegram):
  ```python
  from alerts.llm_summary import armar_callback_resumidor
  from alerts.alert_manager import AlertManager
  alerts = AlertManager(
      log_path="data/alerts.log",
      extra_callback=armar_callback_resumidor(
          llm=None,                # o tu cliente OpenAI
          enviar=tu_funcion_telegram,
      ),
  )
  ```

### 8.3 Seed sintético de 1h (probar forecast SIN tener cámara real corriendo 1h)
```powershell
& ".\.venv\Scripts\python.exe" src/utils/seed_demo_forecast.py
& ".\.venv\Scripts\python.exe" src/prediction/top_riesgo.py
```
Crea 180 filas (1 muestra/min) de 3 repisas con distintas pendientes, genera 4 alertas, luego
`top_riesgo.py` te muestra correctamente a `repisa_yogures` como la de mayor riesgo.

---

## 9. Tests
```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```
Cubre:
- `tests/test_roi_handler.py` (4 tests): rectángulo, polígono, bounds, draw, crop.
- `tests/test_image_enhancer.py` (5 tests): shape, denoise, CLAHE, gamma activo, gamma identity.
- `tests/test_alerts.py` (4 tests): umbral no superado, alerta 1 vez, cooldown, cooldown expira.
- `tests/test_forecast.py` (8 tests): logger CSV, slope, fallback prophet/sarima, orquestación.

## 10. Subir a Git

```powershell
git init
git add .
git status          # mirá que NO aparezcan .venv / data/ / models/ (por el .gitignore)
git commit -m "init tesis: shelf monitor + preproc + YOLO + forecast + alertas + tests"
git branch -M main
git remote add origin <URL_DE_TU_REPO>
git push -u origin main
```

## Conceptos clave / dónde leer más
- Diagrama de flujo oficial del pipeline: `docs/flow.md` (Mermaid, se renderiza en GitHub).
- Notas de kickoff y definiciones cerradas: `kickoff-notes.md`.
  - FURIA = **Fuzzy Unordered Rule Induction Algorithm** (Fürnkranz 2008) — queda anotado como
    capa futura de decisión (no implementado, pendiente de datos etiquetados de alertas).
