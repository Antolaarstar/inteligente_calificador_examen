# modules/turbo_ocr.py

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preparar_imagen_turbo(imagen):
    """
    Preprocesamiento avanzado para OCR: escala, ecualiza, suaviza, binariza.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    ecualizada = cv2.equalizeHist(gris)
    suavizada = cv2.GaussianBlur(ecualizada, (3, 3), 0)

    binarizada = cv2.adaptiveThreshold(
        suavizada, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 9
    )

    # Escalar imagen para mejorar OCR
    escala = 2.0
    ancho = int(binarizada.shape[1] * escala)
    alto = int(binarizada.shape[0] * escala)
    escalada = cv2.resize(binarizada, (ancho, alto), interpolation=cv2.INTER_LINEAR)

    return escalada

def detectar_bloques_manuscritos(imagen_binaria, min_altura=40):
    """
    Detecta áreas grandes de texto manuscrito.
    """
    invertida = cv2.bitwise_not(imagen_binaria)
    contornos, _ = cv2.findContours(invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bloques = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_altura and w > 50:
            bloque = imagen_binaria[y:y+h, x:x+w]
            bloques.append((y, bloque))

    # Ordenar de arriba a abajo
    bloques.sort(key=lambda x: x[0])
    return [b[1] for b in bloques]

def leer_ocr_puro(imagen, idioma='spa', psm=6):
    """
    Lee una imagen con OCR usando config personalizada.
    """
    config = f'--oem 3 --psm {psm}'
    return pytesseract.image_to_string(imagen, lang=idioma, config=config)
