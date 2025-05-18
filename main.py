from modules.image_preprocessor import mejorar_contraste_para_ocr
from modules.ocr_reader import leer_texto

imagen = mejorar_contraste_para_ocr("data/2.jpg")
texto = leer_texto(imagen)

print("Procesando imagen...")
imagen = mejorar_contraste_para_ocr("data/2.jpg")
print("Imagen procesada.")

print("Aplicando OCR...")
texto = leer_texto(imagen)
print("OCR finalizado.")

print("\nTexto detectado:\n")
print(texto)


