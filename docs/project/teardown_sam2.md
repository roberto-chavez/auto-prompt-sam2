# Teardown de SAM 2.1 hiera-tiny (Fase A de E2)

Alcance: **solo el modelo base de E2** (SAM 2.1 tiny, `facebook/sam2.1-hiera-tiny@de431c40…`, Apache-2.0). Este documento reúne en un solo lugar lo que el PDF exige para la Fase A —«conteo de parámetros y FLOPs por bloque, visualización de características intermedias, mapa del campo receptivo, y medición de latencia/memoria … entender por qué el modelo tiene la forma que tiene»— y señala de dónde sale cada cifra. DINOv3 se midió en la etapa 4 pero no es el modelo base de E2 y aquí no se usa.

## 0. Procedencia de cada dato (qué es medido, qué es contado, qué es interpretación)

| Tipo | Origen | Dónde está |
| --- | --- | --- |
| **Medido (nativo, repositorio)** | Etapa 4 (`run_stage4.py`, Colab T4, otra sesión del equipo) | [`evidence/stage4/colab_run_20261029T162558Z/verified_artifacts/`](evidence/stage4/colab_run_20261029T162558Z/verified_artifacts/): `sam_parameters.csv`, `sam_official_0_flops.csv`, `sam_official_0_profile.json`, `sam_official_0_timings.csv`, `sam_official_{0,1}_features.json`, `sam_official_{0,1}_figure.png` (+ `leave_paraguay_out`) |
| **Medido (reejecución propia de E2)** | `E2_01` ejecutado en el Colab del usuario (misma configuración, mismos pines) | Salidas incrustadas, extraídas en [`evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/`](evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/): FLOPs por bloque idénticos (296 051 807 232), latencia 152.6 / 154.9 ms (etapa 4: 148.6 / 151.2 ms), figura de features/ERF |
| **Contado desde la arquitectura** | Instanciación con **pesos aleatorios** en CPU (el conteo no depende de los pesos) | [`teardown_architecture_counts.py`](../../scripts/teardown_architecture_counts.py) → [`evidence/e2/teardown_architecture_counts.json`](evidence/e2/teardown_architecture_counts.json); el script exige que las sumas igualen los totales medidos |
| **Interpretación** | Medidas anteriores + descripción de SAM y SAM 2 en sus artículos | §6; separo lo que dicen los artículos de lo que infiero yo |

## 1. Parámetros y FLOPs por bloque

Total medido: **38 962 498** parámetros, 0 entrenables en la línea base. FLOPs: `torch.utils.flop_counter.FlopCounterMode` (PyTorch 2.7.1), **2 FLOPs por multiply-add**, solo forward, batch 1, SAM a 1024×1024, dos puntos fijos, ruta de imagen estática. **Parciales**: PyTorch solo cuenta operaciones con fórmula registrada.

| Bloque (nombre en el código) | Qué hace en la ruta de imagen | Parámetros (medido) | % parámetros | FLOPs contados | % FLOPs contados |
| --- | --- | ---: | ---: | ---: | ---: |
| `image_encoder` (Hiera + FPN) | Convierte la imagen 1024² en un mapa de características 256×64×64 y dos mapas de mayor resolución | 27 219 136 | 69.86 % | 290 783 394 816 | 98.22 % |
| `sam_prompt_encoder` | Convierte puntos/cajas/máscaras en embeddings dispersos y densos | 6 220 | 0.02 % | 1 536 | ≈ 0 % |
| `sam_mask_decoder` | Transformer de dos capas que cruza tokens de prompt con la imagen y produce 3 máscaras + puntajes | 4 215 109 | 10.82 % | 3 655 700 992 | 1.23 % |
| `memory_attention`, `memory_encoder`, `obj_ptr_proj`, `obj_ptr_tpos_proj` | Memoria de vídeo: **no se llaman** en imagen estática | 5 922 304 + 1 384 608 + 197 376 + 16 448 = 7 520 736 | 19.30 % | `null` (0 llamadas) | — |
| `residual` (tokens/parámetros a nivel raíz; ejecución) | Parámetros sueltos; en FLOPs, `conv_s0/s1` y operaciones fuera de bloques | 1 297 | ≈ 0 % | 1 612 709 888 | 0.54 % |
| **Total** | | **38 962 498** | 100 % | **296 051 807 232** (≈ 296.05 GFLOPs) | 100 % |

Notas: (a) `conv_s0/s1` pertenecen al decoder por parámetros pero se ejecutan dentro de `forward_image`; su costo cae en `residual`, sin duplicarse. (b) Los porcentajes de FLOPs son sobre lo **contado**, no sobre el trabajo real. (c) Fuentes: `sam_parameters.csv`, `sam_official_0_flops.csv`; reproducidos sin diferencia en la reejecución de E2.

### Sub-bloques (contados desde la arquitectura; no hay FLOPs medidos a este nivel)

| Sub-bloque | Parámetros | % del total | Observación |
| --- | ---: | ---: | --- |
| Hiera trunk | 26 849 472 | 68.91 % | `patch_embed` 14 208, embeddings de posición 10 848 y 12 bloques |
| · etapa 0 (bloque 0, dim 96) | 111 840 | 0.29 % | Resolución 256×256 (stride 4) |
| · etapa 1 (bloques 1–2, dim 192) | 852 864 | 2.19 % | Pooling de consultas en el bloque 1 |
| · etapa 2 (bloques 3–9, dim 384) | 12 273 792 | 31.50 % | 7 bloques; atención global en 5, 7 y 9; pooling en el 3 |
| · etapa 3 (bloques 10–11, dim 768) | 13 585 920 | 34.87 % | 2 bloques; pooling en el 10 |
| FPN neck | 369 664 | 0.95 % | Convoluciones 1×1 (768/384/192/96 → 256) |
| Mask decoder · TwoWayTransformer (2 capas, 8 cabezas) | 3 291 264 | 8.45 % | Atención token↔imagen en ambos sentidos |
| Mask decoder · hipernetworks (4 tokens de máscara) | 559 232 | 1.44 % | Generan los pesos que producen cada máscara |
| Mask decoder · cabezas de IoU / de objeto | 132 612 / 131 841 | 0.34 % / 0.34 % | IoU predicho (sirve para escoger máscara) y presencia de objeto |
| Mask decoder · upscaling, `conv_s0`, `conv_s1`, tokens | 73 952; 8 224; 16 448; 1 536 | 0.26 % | Reescalado de la máscara y proyección de características de alta resolución |

## 2. Features intermedias (capa, dimensiones, visualización)
- Capas: `image_encoder.trunk.blocks.0`, `.6` y `.11` (primera, intermedia y última). Formas observadas (entrada 1024²): BHWC **[1,256,256,96]**, **[1,64,64,384]**, **[1,32,32,768]**; se transponen explícitamente a BCHW para visualizar. Registradas en `sam_official_{0,1}_features.json` junto con las formas de todos los bloques ejecutados.
- Visualización: canales fijos 0/1/2 de cada capa (sin PCA ni proyección aprendida), 2 chips de entrenamiento. Figura nativa [`sam_official_0_figure.png`](evidence/stage4/colab_run_20261029T162558Z/verified_artifacts/sam_official_0_figure.png) y la de la reejecución de E2 [`stage4_sam_figure_from_notebook.png`](evidence/e2/colab_run_20261030T013250Z/derived_from_notebooks/stage4_sam_figure_from_notebook.png) (780×1450 px; comprobado que se renderiza y que títulos, rangos min/max y mapas son legibles).
- Lectura descriptiva: el bloque 0 conserva textura fina y una cuadrícula regular; el 6 muestra estructuras más gruesas; el 11 queda en una malla de 32×32 con bandas horizontales. Cada panel usa su propio min/max de color: **no comparar amplitudes entre paneles ni darles significado de «agua»**.

## 3. Campo receptivo efectivo (ERF)
- **Método:** gradiente absoluto, `torch.autograd.grad`, del canal 0 en la posición central de `vision_features` [1,256,64,64] (objetivo [0,0,32,32]) respecto a la entrada normalizada; suma en RGB, normalizado a masa 1; sin `backward()` ni optimizador.
- **Resultado:** la masa del gradiente fuera de la mitad central de la entrada es 38.16 % y 36.53 % en los dos chips (régimen `official`): la sensibilidad no se concentra solo en el centro.
- **Limitaciones:** es **sensibilidad local** para un chip, un canal y una posición, no un campo receptivo teórico ni una prueba causal; está calculado respecto a la entrada normalizada; el panel logarítmico mejora la legibilidad pero no cambia el array. Con dos chips del mismo evento (Ghana) no es representativo.

## 4. Latencia y memoria
Hardware Tesla T4 (15 360 MiB, driver 580.82.07), PyTorch 2.7.1, CUDA 12.6; float32, sin autocast ni TF32; batch 1; entrada 1024² con 2 puntos fijos; **5 warmups y 20 repeticiones** con `cuda.synchronize()` antes y después de cada una; mediana de pared. Medido sobre la ruta tensorial precargada (excluye descarga y carga de pesos, TIFF, transformaciones en CPU, transferencias H2D/D2H, hooks, gradientes, figuras y persistencia).

| Medición | Etapa 4 (equipo) | Reejecución en E2 |
| --- | --- | --- |
| Latencia ruta completa, mediana | 148.6 ms (official) / 151.2 ms (excluido) | 152.6 / 154.9 ms |
| Pico de memoria asignada | 652.713 MiB (delta 288.0 MiB) | 684 419 584 B = 652.7 MiB |
| Replay aislado del encoder / decoder / prompt encoder | — (en `sam_official_0_timings.csv`) | 129.9 / 31.3 / 1.4 ms (no aditivos: replays aislados con entradas precargadas) |

No es el tiempo de un método automático ni incluye E/S. El pico reservado conserva caché del calentamiento y no equivale a la memoria incremental necesaria.

## 5. Operaciones no cubiertas y convención
Lo cubierto son las operaciones con fórmula registrada en `FlopCounterMode` (matmul, convoluciones, atención fusionada y similares). **Sin fórmula** en SAM, según `sam_official_0_profile.json`/`interpretacion_y_cierre.md`: `max_pool2d`, interpolación bicúbica, `sigmoid`, `relu`, `sqrt` y otras; sesgos, normalizaciones, activaciones y softmax pueden faltar o estar solo parcialmente representados. Algunas son vistas o copias sin aritmética y otras sí cuestan; **no se asigna costo cero real** ni se suman filas jerárquicas como costos exclusivos. La atención fusionada puede contar solo una parte del trabajo.

## 6. Por qué el modelo tiene esta forma (interpretación)
Cada punto separa lo que dicen los artículos (SAM: Kirillov et al., 2023; SAM 2: Ravi et al., 2025, ver referencias en el [resumen WAIMLAP](../WAIMLAP/Abstract_CostOfThePrompt.md)) de lo que muestran **nuestras** medidas.

1. **El costo está en el encoder, que corre una vez por imagen.** *Artículos:* el encoder pesado se ejecuta una vez por imagen y su costo se amortiza entre prompts, y el decoder liviano permite prompts interactivos. *Medido aquí:* el encoder aporta 98.22 % de los FLOPs contados y el decoder 1.23 %; el replay aislado del decoder es ~31 ms frente a ~130 ms del encoder. La estructura «encoder pesado / decoder ligero» es coherente con esa lógica; nuestras medidas no prueban la motivación de diseño, solo su consecuencia en costo.
2. **Encoder jerárquico y características multiescala.** *Artículos:* SAM 2 usa Hiera, jerárquico y preentrenado con MAE; las características de menor stride saltan directo al decoder para afilar las máscaras. *Medido/contado aquí:* cuatro etapas con dimensión 96→192→384→768; `conv_s0/s1` proyectan las dos características de mayor resolución hacia el decoder (visible en el `residual` de FLOPs); las dos etapas finales concentran 96.3 % de los parámetros del trunk, pero la resolución espacial baja (64² y 32²) las hace caras en parámetros y baratas por posición. Atención global solo en tres bloques (5, 7, 9) y ventanas en el resto es el mecanismo declarado en la configuración, **no medimos su efecto en FLOPs por bloque**.
3. **Decoder de dos capas con tokens de IoU y tres máscaras.** *Artículos:* tres máscaras para prompts ambiguos (todo/parte/subparte) y una cabeza que estima el IoU para ordenarlas. *Medido aquí:* el decoder tiene 10.82 % de los parámetros y 1.23 % de los FLOPs; las cabezas de IoU y objeto suman ~0.68 %. Nuestra intervención (E2) ajusta justo este bloque por ser el único pequeño y barato, y se declara que las cabezas de IoU/objeto no reciben gradiente con BCE + Dice.
4. **19.30 % de los parámetros no se usa en imagen estática.** SAM 2 es un modelo de imagen **y vídeo**; la memoria (atención y codificador de memoria, punteros de objeto) tiene 7 520 736 parámetros que en nuestra ruta no se llaman (FLOPs `null`). Para segmentar un chip aislado son peso muerto en disco y memoria; no afectan el cómputo medido.
5. **Qué no podemos afirmar.** No medimos por qué Meta eligió estas dimensiones ni comparamos con otras arquitecturas; no hay FLOPs por etapa de Hiera; la latencia es de una sola GPU T4 y una sesión.

*Uso de IA:* esta explicación y el script de conteo fueron redactados con asistencia de Claude a partir de las medidas y de los artículos citados; el estudiante debe poder explicarlos y defenderlos en la defensa oral individual.
