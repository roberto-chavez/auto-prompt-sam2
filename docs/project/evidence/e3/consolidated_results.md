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

| Régimen | Fila etapa 7 (validación) | IoU agua | Precisión | Recall | Dice |
| --- | --- | ---: | ---: | ---: | ---: |
| leave_paraguay_out | a1_assisted | 0.151506 | 0.152129 | 0.973692 | 0.263144 |
| leave_paraguay_out | a1_grid | 0.208799 | 0.905603 | 0.213444 | 0.345465 |
| leave_paraguay_out | a1_points | 0.316842 | 0.354323 | 0.749701 | 0.481215 |
| leave_paraguay_out | a1_whole_box | 0.100493 | 0.101790 | 0.887449 | 0.182632 |
| leave_paraguay_out | a2_negative_0 | 0.282937 | 0.303001 | 0.810348 | 0.441077 |
| leave_paraguay_out | a2_negative_1 | 0.312383 | 0.347956 | 0.753425 | 0.476055 |
| leave_paraguay_out | a2_negative_5 | 0.319942 | 0.360289 | 0.740734 | 0.484782 |
| leave_paraguay_out | a2_positive_1 | 0.292851 | 0.329456 | 0.724954 | 0.453031 |
| leave_paraguay_out | a2_positive_10 | 0.334306 | 0.364692 | 0.800493 | 0.501093 |
| leave_paraguay_out | a2_positive_5 | 0.326940 | 0.360275 | 0.779416 | 0.492772 |
| leave_paraguay_out | a3_frozen | 0.316842 | 0.354323 | 0.749701 | 0.481215 |
| leave_paraguay_out | a3_tuned | 0.505815 | 0.737514 | 0.616865 | 0.671816 |
| leave_paraguay_out | historical_assisted | 0.168434 | 0.169336 | 0.969341 | 0.288307 |
| official | a1_assisted | 0.154020 | 0.155026 | 0.959565 | 0.266927 |
| official | a1_grid | 0.178537 | 0.921761 | 0.181284 | 0.302980 |
| official | a1_points | 0.298615 | 0.327153 | 0.773923 | 0.459897 |
| official | a1_whole_box | 0.097756 | 0.099182 | 0.871778 | 0.178102 |
| official | a2_negative_0 | 0.283212 | 0.292630 | 0.897955 | 0.441411 |
| official | a2_negative_1 | 0.306909 | 0.323078 | 0.859793 | 0.469671 |
| official | a2_negative_5 | 0.312568 | 0.345864 | 0.764530 | 0.476270 |
| official | a2_positive_1 | 0.269888 | 0.298081 | 0.740502 | 0.425059 |
| official | a2_positive_10 | 0.315061 | 0.338254 | 0.821267 | 0.479158 |
| official | a2_positive_5 | 0.310350 | 0.336265 | 0.801080 | 0.473691 |
| official | a3_frozen | 0.298615 | 0.327153 | 0.773923 | 0.459897 |
| official | a3_tuned | 0.484067 | 0.688070 | 0.620159 | 0.652352 |
| official | historical_assisted | 0.158044 | 0.161073 | 0.893660 | 0.272950 |
