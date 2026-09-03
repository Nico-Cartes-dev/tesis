"""
Preprocesamiento de imágenes de cámaras de seguridad de baja calidad.

Cubre:
  - Reducción de ruido (denoising)
  - Mejora de contraste local (CLAHE) -> compensa mala iluminación
  - (Opcional) Super-resolución para cámaras de resolución muy baja

Basado en técnicas estándar de procesamiento de imágenes (Gonzalez & Woods).
No usa deep learning en esta etapa (excepto la super-resolución opcional),
para mantener la latencia baja, ya que esto corre ANTES del modelo de detección.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    denoise_activo: bool = True
    denoise_fuerza: int = 7
    clahe_activo: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    gamma_activo: bool = False
    gamma_valor: float = 1.0   # 1.0 = sin cambio; <1 = aclarar; >1 = oscurecer


class ImageEnhancer:
    """Aplica una cadena de mejoras a un frame antes de pasarlo al modelo."""

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        # Cache LUT gamma (no cambia por frame)
        self._gamma_lut = self._build_gamma_lut(self.config.gamma_valor) \
            if self.config.gamma_activo and self.config.gamma_valor != 1.0 else None

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Corrección gamma (Gonzalez & Woods cap. 3): O = 255 * (I/255)^(1/gamma)."""
        inv = 1.0 / max(1e-3, gamma)
        tabla = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return tabla

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce ruido tipo 'grano' típico de cámaras baratas con poca luz.
        fastNlMeansDenoisingColored es más lento que un gaussian blur,
        pero preserva mucho mejor los bordes (importante para no perder
        la forma de los productos/huecos en la repisa).
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=self.config.denoise_fuerza,
            hColor=self.config.denoise_fuerza,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Mejora contraste POR ZONAS de la imagen en vez de globalmente,
        lo cual es clave en góndolas: la parte de arriba puede estar
        bien iluminada y la de abajo en sombra, en la misma imagen.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        Corrección gamma por lookup-table (muy rápida, no cambia el tamaño).
        - gamma < 1.0 : aclara zonas oscuras (útil en supermercados con luz pobre)
        - gamma = 1.0 : sin cambio
        - gamma > 1.0 : oscurece zonas quemadas
        Referencia: Gonzalez & Woods, Digital Image Processing, capítulo 3.
        """
        if self._gamma_lut is None:
            return image
        return cv2.LUT(image, self._gamma_lut)

    def sharpen(self, image: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """Sharpening suave para compensar la pérdida de nitidez del denoising."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Pipeline completo: denoise -> CLAHE -> gamma -> sharpen leve."""
        result = image.copy()

        if self.config.denoise_activo:
            result = self.denoise(result)

        if self.config.clahe_activo:
            result = self.apply_clahe(result)

        if self.config.gamma_activo:
            result = self.apply_gamma(result)

        result = self.sharpen(result, amount=0.3)

        return result


if __name__ == "__main__":
    # Prueba rápida sobre una imagen de ejemplo
    import sys

    if len(sys.argv) < 2:
        print("Uso: python image_enhancer.py <ruta_imagen>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"No se pudo leer la imagen: {sys.argv[1]}")
        sys.exit(1)

    enhancer = ImageEnhancer()
    processed = enhancer.process(img)

    output_path = sys.argv[1].rsplit(".", 1)[0] + "_processed.jpg"
    cv2.imwrite(output_path, processed)
    print(f"Imagen procesada guardada en: {output_path}")
