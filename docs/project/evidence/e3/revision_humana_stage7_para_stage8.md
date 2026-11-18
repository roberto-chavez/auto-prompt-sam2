# Revisión humana previa a etapa 8

Fecha de preparación: 2026-11-18 (America/Lima). Este documento prepara la revisión; **no la declara completada**. Ningún conjunto reservado se abrió.

## Evidencia que debe revisar el responsable del equipo

- `notebooks/07_ablaciones_y_seleccion_ejecutado.ipynb` — 11 celdas, cuatro celdas de código con salida guardada y ninguna salida de error registrada.
- `docs/project/evidence/e3/revision_stage7_20261118.json` — revisión de ambos regímenes; hashes de ZIP, locks, fallos y cobertura.
- `docs/project/evidence/e3/recovery_20261118.ipynb` — recuperación en un runtime distinto.
- `docs/project/evidence/e3/consolidated_results.md` y `docs/project/cuaderno_decisiones.md` — cifras e interpretación de A1–A3.
- Los dos `evidence.zip` originales y logs de Colab en Drive, si el revisor necesita inspeccionar la salida primaria.

## Hechos que registra la evidencia existente

- Los dos locks computacionales aparecen verificados; ambos regímenes tienen 13 filas, `a3_tuned` seleccionado y cero fallos registrados.
- El informe de revisión registra 1 157 máscaras de validación auditadas para `official` y 975 para `leave_paraguay_out`.
- La recuperación offline aparece presente en ambos regímenes. La salida registrada conserva `reserved_access: []`.
- Hashes reportados para los ZIP: `df51ddc57ea3f9dac31443f1873d5fdcba8fadeb4f264d3245f7cc7c8a62c55b` (`official`) y `89c24a9afb140e24a9335756ba5ef8d21b4fa2f0796d05b1445301e6e48ab665` (`leave_paraguay_out`).
- La salida de esta auditoría sigue siendo “consistencia portátil aprobada, no cierre de etapa”; sus dos entradas de revisión humana están pendientes.

Estos puntos se transcriben de los artefactos registrados en el repositorio. La preparación de este documento no sustituye la inspección humana de los notebooks, logs, configuración de hardware, fallos e interpretación.

## Protocolo que se congelará antes de acceder a reservados

Archivo: `configs/stage8_protocol.json`
SHA-256 canónico calculado: `0ccacde9dbf259d1d4403085e331ef34b93fc2381291638d69cd14a0c3cbc681`

El protocolo contempla evaluación, no ajuste: `test_seen` y el conjunto geográfico reservado por régimen; filas `a1_assisted`, `a1_points` y `a3_tuned`; métricas y bootstrap pareado declarados; sin retocar pesos, umbrales, k ni selección tras abrir reservados. El notebook actual `08_evaluacion_final_geografica.ipynb` todavía no conecta estos predictores ni ejecuta el evaluador.

## Revisión humana de etapa 7 — confirmada por el usuario

- Revisor/a: usuario del proyecto (identidad no inferida).
- Fecha: 2026-11-18.
- Alcance confirmado en la conversación: notebook ejecutado, logs/hardware, fallos e interpretación de etapa 7.
- Registro: [human_review_stage7_user_20261118.json](human_review_stage7_user_20261118.json).
- Autorización para continuar con etapa 8: sí, confirmada en la conversación.

## Revisión del protocolo de etapa 8

- Hash candidato en la fecha de preparación: `0ccacde9dbf259d1d4403085e331ef34b93fc2381291638d69cd14a0c3cbc681`.
- La revisión humana anterior cubrió la evidencia de etapa 7, no el cambio metodológico de etapa 8. Ese cambio añade `a3_frozen` para separar el efecto del decoder del cambio entre la API predictor y la ruta directa. La configuración se corrigió y congeló antes de abrir los conjuntos reservados. Registro: [stage8_protocol_freeze_20261118.json](stage8_protocol_freeze_20261118.json).
- Ningún conjunto reservado se ha abierto.

No completar con datos inferidos. Si se encuentra una discrepancia, corregirla y volver a auditar antes de abrir los reservados.
