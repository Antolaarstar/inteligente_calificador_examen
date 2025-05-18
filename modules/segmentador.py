# modules/segmentador.py

import cv2

def detectar_bloques_respuesta(imagen_cv2, min_altura=40):
    """
    Detecta áreas de texto manuscrito (bloques grandes) en una imagen binarizada.
    Retorna una lista de regiones (como recortes de imagen).
    """
    # Convertir a binaria si no lo está
    if len(imagen_cv2.shape) == 3:
        gris = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen_cv2

    _, binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertimos colores para que el texto sea blanco sobre fondo negro
    invertida = cv2.bitwise_not(binarizada)

    # Buscar contornos
    contornos, _ = cv2.findContours(invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bloques = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtro: solo bloques suficientemente altos (evita líneas pequeñas)
        if h >= min_altura and w > 50:
            bloque = imagen_cv2[y:y+h, x:x+w]
            bloques.append(bloque)

    # Ordenar de arriba a abajo
    bloques_ordenados = sorted(bloques, key=lambda img: cv2.boundingRect(cv2.findContours(cv2.bitwise_not(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])[1])
    
    return bloques_ordenados
