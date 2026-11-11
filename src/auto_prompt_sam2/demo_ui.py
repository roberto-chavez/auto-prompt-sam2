"""Gradio presentation and per-session state. Preview never calls the backend."""
import logging

import gradio as gr

from .demo_backend import DemoBackend
from .demo_config import artifact_contract, configured_decoder, load_demo_config
from .demo_images import LAYOUTS, draw_points, load_image, validate_points

LOG = logging.getLogger(__name__)


class DemoController:
    def __init__(self, config, backend=None):
        self.config = config
        self.report, self.params = artifact_contract(config)
        self.backend = backend if backend is not None else DemoBackend(config)

    @staticmethod
    def cleared(status, image=None):
        # Every input edit clears both old masks, both overlays and their metadata.
        return image, [], image.preview() if image else None, [], None, None, None, None, {}, status

    def upload(self, path, layout):
        if path is None:
            return self.cleared("Carga un chip para comenzar")
        try:
            image = load_image(path, layout, self.params, self.config["max_upload_mb"])
            message = "Imagen cargada. Marca agua (+) y fondo (−)."
            if "original_wh" in image.metadata:
                w, h = image.metadata["original_wh"]
                message += (f" PNG/JPEG {w}×{h} adaptado al lienzo 512×512 sin deformar. "
                            "Uso exploratorio: si es una captura, conviene usar la imagen sin título ni bordes.")
            return self.cleared(message, image)
        except Exception as error:
            LOG.info("Entrada rechazada: %s", type(error).__name__)
            # No stale image can remain after a rejected upload.
            return self.cleared(f"No se cargó la imagen: {error}")

    def edit(self, image, points, action, xy=None, positive=True):
        if image is None:
            return [], None, [], None, None, None, None, {}, "Primero carga una imagen"
        updated = [list(point) for point in points]
        message = "Puntos actualizados; pulsa Segmentar para obtener un resultado nuevo"
        try:
            if action == "clear":
                updated = []
            elif action == "undo":
                updated = updated[:-1]
            else:
                if not isinstance(xy, (tuple, list)) or len(xy) != 2:
                    raise ValueError("Selecciona un píxel de la imagen")
                updated.append([int(xy[0]), int(xy[1]), int(positive)])
                validate_points(updated, image, self.config["max_points"], require_positive=False)
        except ValueError as error:
            updated = [list(point) for point in points]
            message = str(error)
        return updated, draw_points(image, updated), updated, None, None, None, None, {}, message

    def segment(self, image, points, compare):
        if self.config["mode"] == "preview":
            return None, None, None, None, {}, "Modo preparación: imágenes y puntos disponibles; inferencia deshabilitada"
        try:
            validate_points(points, image, self.config["max_points"])
            result = self.backend.segment(image, points, compare)
            frozen, tuned = result["frozen"], result["tuned"]
            metadata = {"input": image.metadata, "points_xy_label": points,
                        "policy": "all_manual_points_in_one_object_prompt",
                        "run_id_decoder": self.report["run_id"] if tuned is not None else None,
                        "note": "Scores predichos por SAM; no son IoU medida ni métricas de agua",
                        "frozen_selection": frozen["selection"],
                        "tuned_selection": tuned["selection"] if tuned is not None else None}
            return (draw_points(image, points, frozen["mask"]),
                    draw_points(image, points, tuned["mask"]) if tuned is not None else None,
                    frozen["mask"].astype("uint8") * 255,
                    tuned["mask"].astype("uint8") * 255 if tuned is not None else None,
                    metadata, result["status"])
        except ValueError as error:
            return None, None, None, None, {}, str(error)
        except Exception:
            LOG.exception("No se pudo segmentar")
            return None, None, None, None, {}, "Inferencia no disponible. Revisa GPU, pesos y logs del servidor"


def build_demo(config=None):
    config = config or load_demo_config()
    controller = DemoController(config)
    mode_note = ("**Modo preparación: no se ejecutan modelos.** Puedes cargar imágenes y marcar puntos."
                 if config["mode"] == "preview" else
                 "**Inferencia GPU bajo demanda.** Los pesos se cargan al pulsar Segmentar.")
    decoder_note = ("Decoder configurado; integridad y carga pendientes hasta la primera inferencia."
                    if configured_decoder(config) else "Decoder ajustado pendiente de incorporar.")
    with gr.Blocks(title="El costo del prompt · Sentinel-2", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# El costo del prompt\nSegmentación asistida de agua superficial · SAM 2.1 tiny")
        gr.Markdown(mode_note + "\n\n" + decoder_note)
        gr.Markdown("1. Carga un GeoTIFF **512×512** o un **PNG/JPEG** (se adapta sin deformar). "
                    "2. Elige positivo o negativo y haz clic. "
                    "3. Pulsa **Segmentar**. Verde (+): agua; rojo (−): fondo. "
                    "Todos los puntos describen una región en una sola llamada.")
        image_state, points_state = gr.State(None), gr.State([])
        layout = gr.Dropdown(LAYOUTS, value=LAYOUTS[0], label="Formato y orden de bandas")
        file = gr.File(label="Imagen satelital (máximo 32 MB por defecto)", type="filepath",
                       file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
        with gr.Row():
            canvas = gr.Image(label="Haz clic para añadir puntos", type="numpy", format="png",
                              interactive=False, height=512)
            with gr.Column():
                polarity = gr.Radio(["Positivo (+)", "Negativo (−)"], value="Positivo (+)", label="Tipo de punto")
                table = gr.Dataframe(headers=["x", "y", "etiqueta (1/0)"], datatype="number",
                                     interactive=False, label="Puntos manuales")
                undo = gr.Button("Deshacer último punto")
                clear = gr.Button("Borrar puntos")
                compare = gr.Checkbox(value=True, label="Comparar con decoder ajustado si está disponible")
                run = gr.Button("Segmentar", variant="primary", interactive=config["mode"] == "inference")
        status = gr.Textbox(label="Estado", value="Carga un chip para comenzar", interactive=False)
        with gr.Row():
            frozen = gr.Image(label="SAM congelado · superposición", format="png", interactive=False)
            tuned = gr.Image(label="Decoder ajustado · superposición", format="png", interactive=False)
        with gr.Accordion("Máscaras binarias (blanco = predicción) y detalles", open=False):
            with gr.Row():
                frozen_mask = gr.Image(label="Máscara congelada", format="png", interactive=False)
                tuned_mask = gr.Image(label="Máscara ajustada", format="png", interactive=False)
            details = gr.JSON(label="Entrada, puntos y selección de candidato")
        gr.Markdown("La fase 1 usó puntos simulados desde etiquetas y seleccionó la mejor época en validación. "
                    "Esta demo usa puntos humanos y no calcula métricas sin etiquetas. "
                    "No demuestra segmentación automática, generalización geográfica ni mejora garantizada. "
                    "Los píxeles sin datos se excluyen de la visualización. Las máscaras descargadas son PNG sin georreferencia.")
        upload_outputs = [image_state, points_state, canvas, table, frozen, tuned, frozen_mask, tuned_mask, details, status]
        edit_outputs = upload_outputs[1:]
        result_outputs = [frozen, tuned, frozen_mask, tuned_mask, details, status]
        # Shared event queue also prevents a slow inference from overwriting a newer edit.
        event_args = dict(concurrency_id="demo", concurrency_limit=1, api_visibility="private")
        file.change(controller.upload, [file, layout], upload_outputs, **event_args)
        layout.change(controller.upload, [file, layout], upload_outputs, **event_args)

        def select(image, points, polarity, evt: gr.SelectData):
            return controller.edit(image, points, "add", evt.index, polarity == "Positivo (+)")

        canvas.select(select, [image_state, points_state, polarity], edit_outputs, **event_args)
        undo.click(lambda im, pts: controller.edit(im, pts, "undo"), [image_state, points_state], edit_outputs, **event_args)
        clear.click(lambda im, pts: controller.edit(im, pts, "clear"), [image_state, points_state], edit_outputs, **event_args)
        compare.change(lambda: (None, None, None, None, {}, "Comparación cambiada; vuelve a segmentar"),
                       [], result_outputs, **event_args)
        run.click(controller.segment, [image_state, points_state, compare], result_outputs, **event_args)
    demo.queue(max_size=16, default_concurrency_limit=1)
    return demo
