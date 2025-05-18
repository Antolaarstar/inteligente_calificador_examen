from modules.turbo_ocr import preparar_imagen_turbo, detectar_bloques_manuscritos, leer_ocr_puro
import cv2

# Cargar imagen original
imagen_original = cv2.imread("data/2.jpg")

# Preprocesamiento turbo
imagen_procesada = preparar_imagen_turbo(imagen_original)

# Leer directamente el texto impreso general
texto_impreso = leer_ocr_puro(imagen_procesada, psm=4)
print("📄 TEXTO IMPRESO DETECTADO:\n", texto_impreso)
print("="*60)

# Detectar bloques manuscritos
bloques = detectar_bloques_manuscritos(imagen_procesada)

# Leer cada bloque con OCR
for i, bloque in enumerate(bloques):
    texto = leer_ocr_puro(bloque, psm=6)
    print(f"✍️ RESPUESTA MANUSCRITA #{i+1}:\n{texto}\n{'-'*50}")
