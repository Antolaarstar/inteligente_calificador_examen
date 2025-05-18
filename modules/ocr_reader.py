# modules/ocr_reader.py

import pytesseract
import cv2

# Si estás en Windows, establece la ruta al ejecutable de Tesseract
# Solo descomenta y edita esta línea si Tesseract no se detecta automáticamente
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def leer_texto(imagen_cv2):
    """
    Usa Tesseract OCR para extraer texto de una imagen procesada (ya en escala adecuada).
    """
    texto = pytesseract.image_to_string(imagen_cv2, lang='spa')  # idioma español
    return texto

print("OCR Reader listo.")

