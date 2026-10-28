# Nota de entrega – WAIMLAp 2026 (Idea 2)

**Título:** The Cost of the Prompt: Automatic DINOv3-Derived Prompts for SAM 2.1 Water Segmentation in Sentinel-2 Imagery
**Estado:** proyecto recién iniciado; no hay resultados experimentales. La sección "Results" se renombró a *Planned Evaluation*.

## Archivos
- `Abstract_CostOfThePrompt.docx` / `.pdf`: 2 páginas A4, una columna, plantilla oficial.
- `Poster_CostOfThePrompt.pptx` / `.pdf`: A0 vertical, 841 × 1189 mm (verificado en el PDF), dos columnas, plantilla oficial.
- `CostOfThePrompt.zip`: solo los dos PDF, ≈1 MB (límite 10 MB).
- El resumen usa Arial en lugar de Aptos para que el render sea idéntico en cualquier equipo.

## Fuentes principales
- Web oficial de WAIMLAp (requisitos y plantillas).
- Repositorios y model cards oficiales de SAM 2/2.1 (Meta) y DINOv3; repositorio de Sen1Floods11 (archivos oficiales de particiones).
- Referencias en APA 7 con DOI, solo artículos de revistas/congresos (10): Bonafilia 2020, Drusch 2012, Kirillov 2023, Li 2024, Liu 2024, Ozdemir 2024, Pekel 2016, Ravi 2025, Tellman 2021, Zhang 2024.

## Correcciones a la propuesta E1
1. Paraguay **no** está reservado como evento held-out: aporta 39/14/14 chips a train/val/test (67). Quien está fuera de las tres particiones es Bolivia (15 chips). Para un test de evento real se propone *leave-Paraguay-out*.
2. SAM 2.1 **no** admite prompts de texto (solo clics, cajas y máscaras). Los prompts de texto son de SAM 3, que además es de acceso restringido y no se usa.
3. La malla automática de SAM produce máscaras sin clase; se añade un puntaje con la probabilidad media de DINOv3 para decidir cuáles son agua.
4. Las particiones oficiales son por chip (no por evento), por lo que las métricas en distribución son optimistas.
5. DINOv3 satelital se preentrenó con RGB a 0.6 m; Sentinel-2 es 10 m. Es una brecha de dominio declarada, no un hecho favorable.
6. Sen1Floods11 etiqueta agua superficial (incluye agua permanente), no solo agua de inundación.
7. Los pesos de DINOv3 son de acceso restringido (licencia propia); SAM 2.1 es Apache-2.0.

## Decisiones metodológicas
Línea base asistida por etiquetas (referencia superior, no desplegable); sonda logística de agua entrenada solo con etiquetas de train (variante sin etiquetas con NDWI); picos = puntos positivos, valles = negativos; una llamada a SAM por componente conexo y unión; intervención en el núcleo = ajustar solo el decodificador de máscaras (BCE + Dice); ablaciones A1–A3 con hipótesis H1–H3 previas; IoU de agua agregado por evento y por chip; chips sin agua fuera del IoU por chip; métricas de cómputo por bloque.

## Limitaciones de la verificación (para ser transparentes)
- Los conteos por evento se leyeron de los CSV oficiales vía WebFetch; `gsutil` no estuvo disponible, y suman 252/89/90 + 15 = 446.
- No confirmé la licencia de Sen1Floods11 ni los IDs exactos de HF de las variantes small/base+/large de SAM 2.1.
- DINOv3 es solo arXiv, así que **no figura en las referencias** por tu indicación de usar solo artículos; sus datos se citan como "model card". Si el profesor acepta preprints, se puede agregar.
- Los requisitos de recursos de cómputo (una GPU de consumo) son una suposición por verificar.
- No se ejecutó ningún entrenamiento ni se descargó el dataset.

## Pendientes por completar (marcadores amarillos/naranjas)
1. Nombres de los 3 autores.
2. Departamento/grupo y afiliación exacta.
3. Correo del autor de correspondencia (resumen y póster).
4. Logos de grupo/proyecto (franja bajo el encabezado del póster; los logos WAIMLAp y PUCP son de la plantilla).
5. Enlace del repositorio y código QR (aún no existe el repo).
6. Recursos de cómputo y financiamiento (Agradecimientos).

La fecha límite es el 23 de octubre. No envié el formulario ni publiqué nada.
