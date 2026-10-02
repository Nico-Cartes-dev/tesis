# Contexto de trabajo para Gemini

Fecha de revisión: 2026-09-30

Este documento resume el estado observado del repositorio Shelf Monitor para continuar el trabajo con Gemini. La documentación y las notas describen objetivos; antes de asumir que algo está terminado, verificar los archivos, configuraciones y artefactos que existen en el repositorio.

## Instrucción para continuar

Ayúdame a desarrollar mi tesis y este proyecto de forma incremental. Primero revisa el código y los datos relacionados con la tarea concreta, conserva las interfaces existentes y no cambies archivos o datos fuera del alcance. Distingue claramente entre comportamiento implementado, resultados medidos e ideas pendientes. No inventes métricas ni des por válidas etiquetas generadas automáticamente sin comprobar su origen. Cuando propongas un cambio, explica cómo verificarlo.

Estoy considerando migrar el entrenamiento y la experimentación a Google Colab. Si trabajamos esa migración, ten en cuenta que los datos y pesos grandes están excluidos de Git y que la demo actual de OpenCV usa ventanas locales.

## Objetivo del proyecto

Shelf Monitor busca detectar falta de stock en estantes de supermercado, principalmente a partir de zonas vacías visibles en la góndola. El sistema también contempla calcular ocupación, generar alertas y estimar tendencias de agotamiento a partir del historial. No busca identificar productos específicos como objetivo principal.

Clases configuradas para detección YOLO:

- `0`: `vacio`
- `1`: `producto`

Objetivos declarados para el pipeline en tiempo real: latencia ideal cercana a 1 segundo y máxima de 5 segundos. Estos objetivos no equivalen a resultados de rendimiento ya medidos.

## Componentes existentes

- `src/preprocessing/image_enhancer.py`: preprocesamiento de imagen con denoise, CLAHE, gamma opcional y mejora de nitidez.
- `src/preprocessing/roi_handler.py`: regiones de interés rectangulares o poligonales con coordenadas normalizadas.
- `src/detection/train.py`: entrenamiento Ultralytics YOLO, fija seeds y guarda corridas bajo `models/runs/<nombre>/`.
- `src/detection/detect.py`: inferencia sobre carpeta, imagen o video; calcula porcentajes por ROI, alertas y forecast. La interfaz de visualización usa ventanas OpenCV y no es directamente utilizable en un notebook Colab.
- `src/alerts/alert_manager.py`: alertas reactivas con umbral, duración/cooldown, log y callback.
- `src/alerts/llm_summary.py`: formato/resumen en lenguaje natural para alertas; el modo online requiere integrar un cliente/API.
- `src/prediction/`: historial CSV, forecast (RollingSlope y alternativas Prophet/SARIMA), servicio y ranking de riesgo semanal.
- `src/utils/`: conversión de anotaciones, visualización de dataset, generación de pseudo-etiquetas de huecos y datos sintéticos para demos.
- `tests/`: pruebas unitarias para alertas, forecast, preprocesamiento y ROI.

## Datos y configuración observados

El dataset YOLO procesado está en `data/processed/` y actualmente tiene:

| Split | Imágenes | Archivos de etiqueta | Cajas clase 0 (`vacio`) | Cajas clase 1 (`producto`) |
|---|---:|---:|---:|---:|
| Train | 48 | 48 | 61 | 7.299 |
| Val | 12 | 12 | 17 | 2.762 |

Hay un fuerte desbalance entre clases. Antes de concluir que el detector de huecos tiene buen desempeño, verificar visualmente el origen/calidad de las etiquetas de clase `0`, las métricas por clase y las predicciones. Las notas de kickoff registraban que faltaban etiquetas manuales de huecos; el estado actual ya contiene algunas etiquetas `0`, pero no se ha confirmado aquí si todas son anotaciones manuales o pseudo-etiquetas.

Configuraciones de dataset:

- `configs/dataset.yaml`: train `images/train`, val `images/val`, raíz `../data/processed` relativa al YAML.
- `configs/dataset_train_v2.yaml`: usa `val_train_v2.txt` como validación; su comentario dice que omite `sm_009` por cajas fuera de imagen. Para experimentos reproducibles, revisar qué configuración corresponde a la corrida y evitar sustituirla sin motivo.
- `configs/cameras.yaml`: define dos ejemplos de cámara/ROI y parámetros globales de preprocesamiento, alertas y forecast. Hay que calibrar esos ROIs y umbrales con imágenes reales antes de tratarlos como configuración de producción.

Las carpetas `data/` (incluidos dataset y CSV) están excluidas de Git por `.gitignore`. También se excluyen pesos y corridas de modelos.

## Modelos y corridas disponibles

Artefactos comprobados en el repositorio:

- `models/runs/stock_huecos_v1/weights/best.pt` y `last.pt` existen; también hay `results.csv`, argumentos y gráficas de entrenamiento.
- `models/runs/stock_detector_v1-2/weights/best.pt` y `last.pt` existen; también hay resultados y gráficas.
- `models/runs/stock_huecos_v2/` existe, pero su carpeta `weights/` está vacía y no se encontró `best.pt`.

No interpretar el nombre de una corrida como prueba de que sea la mejor. Consultar su `args.yaml`, `results.csv`, split usado y métricas por clase antes de comparar modelos. No se calcularon ni transcriben aquí métricas de evaluación.

## Entorno y pruebas

`requirements.txt` declara `ultralytics`, `opencv-python`, `numpy` y `pyyaml`. README menciona Prophet y statsmodels como opciones de forecast, pero no son dependencias obligatorias declaradas en ese archivo.

Se ejecutó `python -m unittest discover -s tests -v` el 2026-09-30. Los cuatro tests de alertas pasaron. Los módulos de tests de forecast, image enhancer y ROI no se pudieron importar porque el Python activo no tiene instalados `numpy` y `opencv-python` (`cv2`); por tanto, la suite completa no está verificada en este entorno. El README documenta 21 tests en total.

## Comandos principales

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

Entrenamiento documentado (el YAML usado debe coincidir con el experimento):

```bash
python src/detection/train.py --data configs/dataset_train_v2.yaml --epochs 80 --batch 8 --nombre stock_huecos_v2
```

Pruebas:

```bash
python -m unittest discover -s tests -v
```

En Windows se puede usar el intérprete del entorno virtual en lugar de `python`. En Colab/Linux, cambiar rutas y visualización; no usar `cv2.imshow` esperando que aparezca una ventana de escritorio.

## Pendientes que conviene priorizar

1. Auditar visualmente las 61 etiquetas de huecos de train y las 17 de val; confirmar si son anotaciones fiables, pseudo-etiquetas o una mezcla.
2. Evaluar `stock_huecos_v1` y `stock_detector_v1-2` con el mismo split, reportar métricas por clase y revisar la matriz de confusión; no concluir por una sola métrica agregada.
3. Aclarar el estado de `stock_huecos_v2`: la carpeta existe, pero no hay checkpoint `best.pt` disponible actualmente.
4. Completar la ejecución de tests después de instalar las dependencias en un entorno compatible.
5. Si se migra a Colab, preparar una ruta de subida/copia del proyecto y dataset desde Drive, usar GPU cuando esté disponible, validar rutas del YAML y mostrar resultados con notebook/Matplotlib. Guardar checkpoints y métricas en Drive porque el almacenamiento local de la sesión Colab es temporal.
6. Probar ROIs, umbrales de alerta y latencia con imágenes/cámaras reales; las configuraciones actuales incluyen valores de ejemplo.

## Estructura relevante

```text
configs/       YAML de dataset y cámaras
data/          dataset procesado, etiquetas, historial y alertas (ignorado por Git)
docs/          flujo del sistema y documentación de ranking
models/runs/   checkpoints y resultados de entrenamiento (ignorado por Git)
src/alerts/    gestión y resumen de alertas
src/detection/ entrenamiento YOLO e inferencia
src/prediction/forecast e historial de stock
src/preprocessing/ mejora de imagen y ROIs
src/utils/     conversión, visualización y herramientas de dataset
tests/         pruebas unitarias
```