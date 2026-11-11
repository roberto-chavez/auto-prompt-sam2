| Régimen | Fila (validación) | IoU agua | Precisión | Recall | Dice |
| --- | --- | ---: | ---: | ---: | ---: |
| official | Etapa 3: puntos desde etiquetas (1+/1− por componente), decoder congelado | 0.158044 | 0.161073 | 0.893660 | 0.272950 |
| official | Etapa 5: puntos automáticos DINOv3 (≤3+/3−), decoder congelado | 0.298615 | 0.327153 | 0.773923 | 0.459897 |
| official | Etapa 5: malla SAM 32×32 puntuada con DINOv3, decoder congelado | 0.178537 | 0.921761 | 0.181284 | 0.302980 |
| official | Etapa 6: decoder entrenado con puntos automáticos, mejor época 10 | 0.484067 | 0.688070 | 0.620159 | 0.652352 |
| leave_paraguay_out | Etapa 3: puntos desde etiquetas (1+/1− por componente), decoder congelado | 0.168434 | 0.169336 | 0.969341 | 0.288307 |
| leave_paraguay_out | Etapa 5: puntos automáticos DINOv3 (≤3+/3−), decoder congelado | 0.316842 | 0.354323 | 0.749701 | 0.481215 |
| leave_paraguay_out | Etapa 5: malla SAM 32×32 puntuada con DINOv3, decoder congelado | 0.208799 | 0.905603 | 0.213444 | 0.345465 |
| leave_paraguay_out | Etapa 6: decoder entrenado con puntos automáticos, mejor época 8 | 0.505815 | 0.737514 | 0.616865 | 0.671816 |
