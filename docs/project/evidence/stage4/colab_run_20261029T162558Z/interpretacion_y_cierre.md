# Etapa 4: interpretación y cierre de la evidencia Colab

Fecha: 2026-10-29 (America/Lima). Run `20261029T162558Z_bfb5852907_c4c91701`. **Etapa 4 Concluida; 4/9 etapas concluidas.** No se inician etapas 5–9 ni demo.

## Evidencia y alcance de la revisión

Se revisaron los originales del usuario sin ejecutar su código: notebook con salidas y ZIP de 220 371 841 bytes. El ZIP coincide con el SHA-256 mostrado por recover. Se verificaron 49 generaciones, 39 artefactos, snapshot de fuentes coincidente y el único cambio del notebook: completar el SHA del bundle. Sus cuatro imágenes incrustadas coinciden byte a byte con los PNG del ZIP; se inspeccionaron visualmente las ocho figuras. `independent_checks.json` verifica shapes de los ocho conjuntos NPZ y recalcula masas ERF desde arrays guardados. No se ejecutaron modelos, extracción ni gradientes en la PC.

Prepare y recover acreditan Tesla T4 (15 360 MiB nominales, capacidad 7.5), driver 580.82.07, PyTorch 2.7.1 y transformers 4.56.2; boot IDs distintos. 50 pruebas y pip check correctos en ambas sesiones, 50 pines base verificados. Todos los parámetros congelados y módulos eval, sin gradientes acumulados; hashes completos de parámetros/buffers iguales antes/después y tras recarga offline, hooks eliminados y repetición de features/ERF dentro de atol=1e-5/rtol=1e-4. El warning de deprecación torch_dtype no impidió cargar float32.

Únicos chips leídos: `Ghana_103272` y `Ghana_1089161`, train en ambos regímenes. Manifiestos y estadísticas propios, generaciones/hashes iguales a etapa 2. Cero etiquetas, validación, test, Bolivia o Paraguay reservado; ninguna proyección aprendida, probe, decoder entrenado, prompt automático ni métrica de agua. Los dos chips son del mismo evento: no hay evidencia representativa de todo train ni comparación geográfica de calidad.

## Medidas observadas

Batch 1, float32, sin autocast/TF32; SAM 1024, DINO 512, dos puntos fijos, 5 warmups y 20 repeticiones con sincronización GPU. Primer chip train por régimen. GFLOPs parciales, 2 operaciones por multiply-add. Memoria de forward después del calentamiento, no pico de toda la disección/ERF.

| Régimen | Modelo | GFLOPs contados | Mediana ms | Pico allocated MiB | Delta allocated MiB | Pico reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| official | sam | 296.051807 | 148.596 | 652.713 | 288.000 | 2090.000 |
| official | dino | 727.194108 | 241.929 | 1270.051 | 45.695 | 2416.000 |
| leave_paraguay_out | sam | 296.051807 | 151.183 | 652.713 | 288.000 | 2090.000 |
| leave_paraguay_out | dino | 727.194108 | 262.670 | 1270.051 | 45.695 | 2416.000 |

La latencia incluye solo la ruta tensorial precargada: en SAM encoder, conv_s0/s1, reshape/no_mem, prompts/PE, decoder y resize de los tres logits; en DINO embeddings/RoPE, 24 capas y norma final. Excluye pesos/descarga, TIFF, transforms CPU, transferencias, hooks, gradientes, figuras y persistencia. No es el tiempo completo de un método automático. El replay por bloque es aislado y no aditivo; todas sus muestras están en validation.json y sus CSV. Reserved conserva caché del calentamiento y no equivale al incremento necesario del forward.

Parámetros SAM: **38 962 498**, suma exacta de image_encoder 27 219 136; prompt encoder 6 220; mask decoder 4 215 109; memory_attention 5 922 304; memory_encoder 1 384 608; obj_ptr_proj 197 376; obj_ptr_tpos_proj 16 448; residual 1 297. Los módulos de memoria/proyecciones de objetos no se llaman en esta ruta, sus tiempos y FLOPs son null, conservando sus parámetros. DINO: **303 129 600** = embeddings 793 600 + 24 × 12 597 248 + norm 2 048; RoPE y residual 0 parámetros. Cero entrenables en ambos.

SHA-256 pesos SAM: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`, igual al usado en etapa 3. DINO: `4e6356d992c1301b5e7c275f465e47296c5c4ad17052e262b29fc21e82ccc698`, revisión autorizada f692fa42da72c6797b67cd73494a168d1120d3ee. Archivos de pesos permanecen en Drive; no forman parte de la evidencia compartida.

## Arquitectura, features y campo receptivo

Hiera bloques 0/6/11: BHWC [1,256,256,96], [1,64,64,384], [1,32,32,768]; visualización BCHW por transposición explícita. El tensor SAM usado para ERF es [1,256,64,64], objetivo [0,0,32,32]. La ruta diferenciable coincidió con el predictor oficial reutilizado en los cuatro casos.

DINO capas 0/11/23 y salida final: [1,1029,1024]. CLS índice 0, cuatro registros [1,5), 1024 parches [5,1029), reshape por filas a [1,1024,32,32]. ERF en [0,0,16,16] del grid después de LayerNorm. Centro de parche nativo (16*col+7.5,16*fila+7.5). No PCA, scaler ni ajuste de pesos.

Inspección visual de los ocho PNG: SAM conserva textura fina en bloque 0 y presenta estructuras más gruesas en 6/11; DINO muestra contrastes espaciales distintos por canal y capa y focos separados en ERF. No se observan mapas vacíos ni gradientes nulos. Los colores usan min/max propios de cada panel y ERF logarítmico: no comparar amplitudes por color entre capas/régimen ni interpretarlas como probabilidad de agua. Algunos títulos largos son densos; nombres y shapes completos están en JSON. No se infieren etiquetas de agua de estas imágenes.

| Régimen | Chip | Masa ERF fuera de la mitad central SAM | DINO |
| --- | --- | ---: | ---: |
| official | Ghana_103272 | 38.16% | 67.95% |
| official | Ghana_1089161 | 36.53% | 55.92% |
| leave_paraguay_out | Ghana_103272 | 38.19% | 70.20% |
| leave_paraguay_out | Ghana_1089161 | 36.53% | 48.24% |

Las ocho masas totales coinciden con 1 dentro del redondeo float32. Esta masa describe gradiente absoluto del canal/posición elegidos respecto a entradas normalizadas. No es campo teórico, importancia causal, atención ni gradiente sobre reflectancia original. Los regímenes usan las mismas imágenes con recortes radiométricos distintos: las diferencias de mapas no son efecto demostrado de excluir un evento.

## Contraste de hipótesis y límites

**H4a respaldada para aritmética cubierta:** el image encoder aporta 98.2204% del conteo SAM; mask decoder 1.2348%, residual de ejecución 0.5447% y prompt encoder el resto diminuto. Las conv_s0/s1 pertenecen al decoder por parámetros, pero se llaman desde forward_image y su costo aparece en residual, sin duplicación. Módulos de vídeo presentes no implican llamadas en imagen estática.

**H4b compatible con estas observaciones:** DINO conserva 48.24–70.20% de masa fuera de la mitad central y focos espaciales no uniformes, frente a 36.53–38.19% en SAM. Es evidencia local para dos chips y un canal/objetivo por modelo; no demuestra superioridad de features ni que todas las cabezas/capas atiendan globalmente de la misma manera.

**H4c respaldada de forma descriptiva:** el cociente de FLOPs contados DINO/SAM es 2.4563, mientras el de latencia es aproximadamente 1.63–1.74. RoPE/norma tienen cero FLOPs registrados y tiempo positivo (official ~0.195/~0.096 ms), una muestra concreta de cobertura incompleta. Esto no aísla causalmente tráfico o kernels; resoluciones, arquitecturas y salidas difieren. Un solo protocolo/medición por régimen no estima variabilidad entre sesiones ni significancia.

Cobertura parcial explícita: DINO no tiene fórmulas contadas para native_layer_norm, GELU, sin/cos, mul/add y operaciones de estructura de tensores, entre otras; SAM añade max_pool2d, interpolación bicúbica, sigmoid, relu, sqrt y otras. Los listados exactos por bloque están en los JSON. Algunas son vistas/copias sin aritmética, otras sí tienen trabajo; no asignar costo verdadero cero ni sumar filas jerárquicas como costos exclusivos. Fused attention también puede contar solo una parte del trabajo.

## Decisión, archivos y pendientes

Se cubrieron parámetros, FLOPs, features, ERF y latencia/memoria con config/hardware/cobertura, persistencia y recuperación revisadas. Se cierra exclusivamente etapa 4. Los campos pending_external_review de los informes originales se preservan y se resuelven mediante review.json y esta interpretación, sin modificar resultados ni repetir modelos en PC.

Original ZIP conservado en `artifacts/stage4-evidence/20261029T162558Z_bfb5852907_c4c91701.zip` (ignorado por Git por tamaño), con SHA-256 y tamaño en review.json. Notebook original, reportes, CSV/figuras y auditorías se conservan en este directorio. La carpeta colab/ queda ignorada para próximas entregas; se retiran sus duplicados tras verificar hashes.

No falta evidencia técnica para esta etapa. Pendientes transversales: respuesta del publicador sobre licencia de datos y revisión/comprensión académica del equipo. No se inicia probe, decoder, prompts automáticos, reservados ni demo. IA: Codex ejecutó auditorías CPU de archivos, contrastó fuentes, revisó las ocho figuras e interpretó las medidas; sin subagentes, modelos locales ni código adjunto ejecutado.
