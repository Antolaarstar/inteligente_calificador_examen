def mejorar_contraste_para_ocr(ruta_imagen, mostrar=False):
    import cv2
    import numpy as np

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise FileNotFoundError(f"No se encontró la imagen en {ruta_imagen}")

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Suavizado más fuerte para eliminar ruido fino
    suavizado = cv2.medianBlur(gris, 5)

    # Umbral adaptativo pero SIN invertir el color
    binarizada = cv2.adaptiveThreshold(
        suavizado, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Opcional: morfología para limpiar puntos pequeños
    kernel = np.ones((2, 2), np.uint8)
    limpia = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)

    if mostrar:
        cv2.imshow("Preprocesada", limpia)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return limpia
