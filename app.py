import os
import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from inference.models import YOLOWorld

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model


MARKDOWN = """
# YOLO-World + EfficientViT-SAM
Powered by Roboflow [Inference](https://github.com/roboflow/inference) and [Supervision](https://github.com/roboflow/supervision) and [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)
"""

# Load models
# yolo_world = YOLOWorld(model_id="yolo_world/s")
# yolo_world = YOLOWorld(model_id="yolo_world/m")
yolo_world = YOLOWorld(model_id="yolo_world/l")
# yolo_world = YOLOWorld(model_id="yolo_world/x")

# yolo_world = YOLOWorld(model_id="yolo_world/v2-s")
# yolo_world = YOLOWorld(model_id="yolo_world/v2-m")
# yolo_world = YOLOWorld(model_id="yolo_world/v2-l")
# yolo_world = YOLOWorld(model_id="yolo_world/v2-x")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = EfficientViTSamPredictor(
    create_sam_model(name="xl1", weight_url="./weights/xl1.pt").to(device).eval()
)


# Load annotators
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def detect(
    image: np.ndarray,
    query: str,
    confidence_threshold: float,
    nms_threshold: float,
) -> np.ndarray:
    # Preparation.
    categories = [category.strip() for category in query.split(",")]
    yolo_world.set_classes(categories)
    # print("categories:", categories)

    # Object detection
    results = yolo_world.infer(image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )
    # print("detected:", detections)

    # Segmentation
    sam.set_image(image, image_format="RGB")
    masks = []
    for xyxy in detections.xyxy:
        mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
        masks.append(mask.squeeze())
    detections.mask = np.array(masks)
    # print("masks shaped as", detections.mask.shape)

    # Annotation
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    labels = [
        # f"{categories[class_id]}: {confidence:.2f}"
        f"{categories[class_id]}: {confidence:.3f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="numpy", label="Input Image")
            image_categories_text_component = gr.Textbox(
                placeholder="you can input multiple words with comma (,)",
            )

            with gr.Accordion("YOLO-WORLD", open=False):
                confidence_threshold_component = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.005,
                    step=0.01,
                    label="Confidence Threshold",
                )

                iou_threshold_component = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label="NMS Threshold",
                )

        yolo_world_output_image_component = gr.Image(type="numpy", label="Output image")
    submit_button_component = gr.Button(value="Submit", scale=1, variant="primary")
    gr.Examples(
        fn=detect,
        examples=[
            [
                os.path.join(os.path.dirname(__file__), "examples/livingroom.jpg"),
                "table, lamp, dog, sofa, plant, clock, carpet, frame on the wall",
                0.05,
                0.5,
            ],
            [
                os.path.join(os.path.dirname(__file__), "examples/cat_and_dogs.jpg"),
                "cat, dog",
                0.2,
                0.5,
            ],
        ],
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
        ],
        outputs=yolo_world_output_image_component,
    )

    submit_button_component.click(
        fn=detect,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
        ],
        outputs=yolo_world_output_image_component,
    )

demo.launch(debug=False, show_error=True)
