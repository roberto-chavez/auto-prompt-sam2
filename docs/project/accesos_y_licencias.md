# Accesos y licencias — fase 1, etapa 1

Consulta: **2026-10-28 (America/Lima)**. **Estado actual:** HEAD autenticado de SAM/DINOv3 autorizado y verificado mediante los informes Colab; solo permanece pendiente la licencia de Sen1Floods11. La tabla inicial conserva el resultado anónimo histórico. Evidencia ejecutada: [public_sources.json](evidence/stage1/public_sources.json), regenerable mediante `python scripts/audit_public_sources.py`. Solo consultas de metadatos, documentación y HEAD; no se adquirieron imágenes, etiquetas, splits ni pesos. Las revisiones se fijan en [environment.yaml](../../configs/environment.yaml).

| Recurso | Identidad confirmada | Acceso comprobado | Licencia y resolución pendiente |
| --- | --- | --- | --- |
| Sen1Floods11 | v1.1, bucket `sen1floods11`; documentación `cloudtostreet/Sen1Floods11@b9e460dabe70230111a3c3d36cd0c5ce03606ee9` | Catálogos públicos y HEAD de un objeto S2Hand/LabelHand, sin leer su cuerpo | Ambos catálogos oficiales declaran `proprietary`. Falta confirmar con el publicador qué términos permiten adquirir S2Hand y LabelHand, entrenar/evaluar y publicar derivados. Bloquea adquisición de etapa 2. |
| SAM 2.1 tiny | `facebook/sam2.1-hiera-tiny@de431c4043854a71d8101e17995dfe596bf101a5`, archivo `sam2.1_hiera_tiny.pt` | HEAD anónimo 302, sin seguir URL firmada; archivo y revisión presentes | Apache-2.0 según ficha y repositorio oficial. Conservar licencia/avisos en cualquier redistribución aplicable. No se redistribuyen pesos aquí. |
| SAM, implementación original | `facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4`, config `configs/sam2.1/sam2.1_hiera_t.yaml` | Commit y setup oficiales consultados | Apache-2.0; extensión opcional de componentes conectados tiene aviso BSD separado. Se desactiva su compilación en esta etapa y se documenta la limitación. |
| DINOv3 ViT-L/16 SAT-493M | `facebook/dinov3-vitl16-pretrain-sat493m@f692fa42da72c6797b67cd73494a168d1120d3ee`, `model.safetensors` | Metadatos públicos confirmados; `gated=manual`; HEAD anónimo **401** | DINOv3 License, no Apache/MIT. Autorización del usuario **no comprobada**. El titular debe aceptar condiciones, recibir aprobación y dar acceso de lectura al token del secreto Colab `HF_TOKEN`. Repetir HEAD hasta respuesta autorizada. |

El SHA de documentación de Sen1Floods11 **no identifica los bytes del dataset**. GCS no ofrece aquí un commit global inmutable: los metadatos de los objetos consultados conservan `generation`, tamaño y hashes publicados, que no son hashes calculados tras descarga. Los SHA-256 locales de pesos y datos permanecen `null`; deberán calcularse cuando su adquisición esté autorizada en la etapa correspondiente. No se sustituyen modelos ni versiones del dataset.

Fuentes primarias:

- [README de Sen1Floods11 en el commit fijado](https://github.com/cloudtostreet/Sen1Floods11/blob/b9e460dabe70230111a3c3d36cd0c5ce03606ee9/README.md), [estructura oficial](https://github.com/cloudtostreet/Sen1Floods11/blob/b9e460dabe70230111a3c3d36cd0c5ce03606ee9/docs/README.md). El bucket correcto es `sen1floods11`; la línea `senfloods11` del README es inconsistente con sus propios ejemplos.
- [Colección fuente](https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_source/collection.json) y [colección de etiquetas](https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_label/collection.json). Sus hashes de metadatos están en la evidencia. La [incidencia de licencia #18](https://github.com/cloudtostreet/Sen1Floods11/issues/18) sigue sin una concesión de licencia verificable del publicador. No se adopta la licencia de un espejo ni de otro dataset de Floodbase.
- [SAM: ficha de pesos](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/de431c4043854a71d8101e17995dfe596bf101a5), [licencia del código](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE).
- [DINOv3: ficha y solicitud de acceso](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m), [licencia oficial consultada](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md). La licencia exige reconocimiento en publicaciones de investigación y condiciones para redistribuir; el titular debe revisar el texto completo. No se aceptaron términos en su nombre.

El catálogo público y la capacidad técnica de descargar no equivalen a una autorización de uso. El bloqueo de Sen1Floods11 se resuelve con una fuente oficial o permiso documentado del publicador que cubra las modalidades y usos anteriores; no basta alterar `license_status`. El 401 de DINOv3 es una prueba anónima, no una afirmación sobre los permisos que pueda tener la cuenta del equipo.

## Actualización de aprobación comunicada por el usuario

El 2026-10-28 el usuario indicó que recibió aprobación para DINOv3 y aportó un notebook de Colab con recover exitoso. Se registra como aprobación comunicada por el titular. El adjunto no contiene la salida de `check_access.py`; este script devuelve 0 incluso si hay bloqueos. Falta leer `access.json` guardado en Drive para acreditar el HEAD autenticado del checkpoint/revisión fijados. La prueba anónima 401 inicial conserva su valor histórico y no contradice una aprobación posterior. La licencia de Sen1Floods11 sigue sin resolver.

## Accesos autenticados confirmados mediante access.zip

Se recibieron y verificaron dos generaciones completas del informe de acceso (22:10:39 y 22:25:52 UTC del 2026-10-28). Ambas tienen SHA-256, tamaño e identidad íntegros, coincidentes con el run `20261028T221055Z_1d0f6a92f4_aa07b10e`. [Último informe](evidence/stage1/colab_run_20261028T221055Z/access.json) y [auditoría](evidence/stage1/colab_run_20261028T221055Z/access_archive_review.json).

- SAM: `facebook/sam2.1-hiera-tiny@de431c4043854a71d8101e17995dfe596bf101a5`, archivo `sam2.1_hiera_tiny.pt`: metadata_matches=true, HEAD 302, authorized=true.
- DINOv3: `facebook/dinov3-vitl16-pretrain-sat493m@f692fa42da72c6797b67cd73494a168d1120d3ee`, archivo `model.safetensors`: metadata_matches=true, HEAD 302, authorized=true.

En ambas comprobaciones body_downloaded=false y no se siguieron redirecciones. Queda confirmado el acceso de la cuenta en las fechas del informe; no se incluyen tokens ni se garantiza acceso perpetuo. El único elemento de blockers es la licencia pendiente de Sen1Floods11, cuyos catálogos siguen declarando proprietary. La prueba anónima inicial 401 de DINOv3 queda como antecedente superado por esta evidencia autenticada.

## Consulta enviada y cierre de seguimiento con salvedad

Se revisó el correo saliente aportado por el usuario, dirigido a support@floodbase.com el 2026-10-28 a las 18:05:23 -05:00. [Registro](evidence/stage1/licensing_request_sent.md). Solicita licencia o permiso para los usos previstos de S2Hand y LabelHand. No contiene una respuesta ni una autorización del publicador.

Por instrucción expresa del usuario, la etapa 1 se cierra con salvedad y se reabrirá al recibir respuesta. La licencia conserva su estado pendiente; no se alteran los informes técnicos ni se declara que el silencio conceda derechos. La instrucción cambia el estado de seguimiento, no los términos de terceros ni la evidencia de autorización.
