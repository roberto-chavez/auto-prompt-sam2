# Demo manual con pesos reales

Roberto ejecutó el backend de la demo en Google Colab el 11 de noviembre de 2026 y envió el notebook y outputs.zip por WhatsApp. La revisión del 18 de noviembre comprobó el ZIP, las salidas del notebook y la máscara original.

- GPU Tesla T4 y SAM 2.1 tiny congelado.
- Chip de validación Ghana_1033830, 512 × 512, 262 144 píxeles válidos.
- Puntos manuales (256, 256, positivo) y (100, 100, negativo).
- Máscara binaria con 23 252 píxeles positivos.
- La selección usa el mayor IoU predicho por SAM. El valor 0.6578 corresponde a confianza predicha y carece de comparación con etiquetas en esta prueba.
- El notebook termina con OK e3_demo_smoke y contiene cero salidas de error.

Esta evidencia verifica carga de pesos e inferencia del backend manual sobre un chip. El recorrido interactivo completo de Gradio, el decoder ajustado y los prompts automáticos quedan fuera de esta ejecución.

El notebook publicado conserva código y salidas de texto. Se retiró una salida con la visualización del chip para conservar los raster y derivados en el Drive privado mientras se aclara la licencia. Los hashes de los originales constan en verification.json. La máscara original y el notebook íntegro permanecen en la carpeta privada de salidas del espacio de trabajo y en el Drive del equipo.

[Salidas originales en Drive](https://drive.google.com/drive/folders/13C3_2fUU9T-EADIC36BmsJ2fe-rxbdvp)
