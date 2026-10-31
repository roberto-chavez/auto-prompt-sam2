# Revisión parcial de etapa 5 — 2026-10-30

Run `20261029T194653Z_8073c068fa_ed6583db`. Notebook original y JSON legibles conservados. ZIP original permanece en `colab_temp/`, sin modificación.

Los reportes y salidas registran prepare/recover completos en T4 y boot IDs distintos, sin reentrenamiento durante recover. Ambos regímenes seleccionan C=0.1, ponderación ninguna, umbral positivo 0.7 y umbral de malla 0.5. IoU val reportada: puntos 0.298615/0.316842; malla 0.178537/0.208799 (official/excluido). Estas cifras aún no están validadas mediante auditoría independiente de todas las predicciones.

El notebook conserva salidas pero tiene contadores nulos en prepare/resumen. El auditor estricto se detuvo por esa condición; se hizo una segunda revisión en memoria permitiendo salidas presentes, sin alterar notebook ni fuentes experimentales. Esa revisión se detuvo por integridad del ZIP: encabezados/directorio inconsistentes y entradas ilegibles. Véanse external_review.json y full_crc_scan.json. El SHA recibido difiere del impreso al finalizar recover; no se conoce todavía si la alteración ocurrió durante descarga/sincronización o en otro paso. No se intenta reparar ni sustituir evidencia.

Acción requerida: descargar de nuevo el ZIP ya terminado y comprobarlo. No se solicita repetir prepare/recover ni consumir GPU para esa descarga. Si el ZIP de Drive también está dañado, exportar nuevamente desde las generaciones originales, sin entrenamiento ni inferencia. Etapa 5 en curso, 4/9 cerradas; NDWI aplazado, etapas posteriores sin iniciar. La demora previa no puede atribuirse con certeza a una operación: faltan tiempos por bloque de recuperación.

IA: Codex inspeccionó notebook/JSON y ejecutó auditoría de transporte CPU, sin modelos, entrenamiento ni datos reservados. Revisión integral pendiente de transporte íntegro.
