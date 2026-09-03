# Notas de Kickoff — Proyecto Shelf Monitor

Notas tomadas en la reunión inicial, organizadas por tema.

## Objetivo del proyecto

- Detectar **falta de stock** en góndola (no identificación de producto específico)
- Enfoque en **detectar la forma** (silueta del hueco vacío en la repisa) más que reconocer el producto exacto

## Computer Vision / Preprocesamiento

- Uso de **Computer Vision** con modelos de Machine Learning
- Imágenes vienen de **cámara de seguridad** con resolución no ideal → requiere preprocesamiento
- **Algoritmo de eliminación de ruido** (denoising)
- **Data Augmentation**
- Libro de referencia: **Gonzalez & Woods** (*Digital Image Processing*)
- "Algoritmo furia" — *sin confirmar, posible confusión con Fourier u otro término. Pendiente de aclarar.*
- No todas las imágenes son iguales (distintas cámaras, ángulos, iluminación)

## Modelos / Predicción

- Posible uso de **modelos de predicción** además de detección
- **Aprendizaje continuo**: el modelo debería poder reentrenarse/mejorar con el tiempo

## Zona de interés (ROI)

- Hay que tomar una **zona específica** de la cámara (la repisa/estante)
- Esa zona **no siempre tiene el mismo tamaño** → necesita manejo dinámico/configurable

## Alertas

- Latencia objetivo: **~1 segundo** (ideal)
- Latencia máxima aceptable: **~5 segundos** (media)
- Definir **criterio de alarma** (qué condición dispara la alerta)

## Documentación e ingeniería

- **Gamma spec** — *ahora definido:* Corrección gamma estándar del procesamiento digital de imágenes (Gonzalez & Woods, cap. 3): `O(x,y) = c · I(x,y)^γ`. Se implementó como etapa opcional en `src/preprocessing/image_enhancer.py` (LUT de 256 entradas, < 1ms por frame) y se configura desde `configs/cameras.yaml` sección `preprocesamiento.gamma`.
  - Valores típicos: `gamma_valor < 1` aclara pasillos oscuros; `gamma_valor > 1` atenúa reflejos quemados.
  - Estado: activo=False por defecto (no altera resultados hasta calibrar).
  - Implementación: `src/preprocessing/image_enhancer.py:84-94` (método `apply_gamma`) + `configs/cameras.yaml:45-52`.
- **Diagrama de flujo**
- **Resultados replicables**
- Buenas prácticas de código — referencia a **Refactoring Guru** (refactoring.guru)
- Uso de **LLM** en algún punto del proyecto — *rol propuesto:* Generar resúmenes en lenguaje natural de cada alerta para el encargado de góndola / repositor. Salida ejemplo: *"La repisa de yogures (cámara 02) lleva 4 minutos con 58% de vacío; en base al historial de las últimas 3 semanas se recomienda reponer antes de las 19:00 hs"*. Ver esqueleto: `src/alerts/llm_summary.py`.

## Pendientes / a confirmar

- [x] **Confirmado ✓** Confirmar qué es "algoritmo furia" → **FURIA = Fuzzy Unordered Rule Induction Algorithm** (Fürnkranz et al, 2008). Clasificador de reglas borrosas sin orden (generalización de RIPPER/JRip). **Dónde encaja en este proyecto:** NO es de visión, sino de TOMA DE DECISIÓN de alerta: sustituye el umbral hardcodeado `si pct_vacio >= X AND tiempo >= Y` por un clasificador que aprende reglas humanamente legibles desde el histórico de alertas. Queda para capítulo de experimentación / futuro trabajo (no se implementa aún). Ver apunte inicial `docs/furia.md` (vacío por ahora, para llenar en ese capítulo).
- [x] Definir "Gamma spec" → hecho, ver sección Documentación e ingeniería arriba.
- [x] Definir rol exacto del LLM en el pipeline → hecho: resumen en lenguaje natural de alertas.
- [x] Confirmar si el dataset ya está etiquetado o hay que etiquetarlo → los datasets `Supermarket shelves` y `dataset 2` ya venían etiquetados y fueron convertidos a formato YOLO (clase `1 = producto`). Queda pendiente agregar ~50-100 etiquetas de clase `0 = vacio` (huecos reales) para cumplir el objetivo de "detectar la forma del hueco".
