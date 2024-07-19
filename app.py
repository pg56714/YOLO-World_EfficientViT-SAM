import os
import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from typing import List
from inference.models import YOLOWorld

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model


MARKDOWN = """
# YOLO-World + EfficientViT-SAM
Powered by Roboflow [Inference](https://github.com/roboflow/inference) and [Supervision](https://github.com/roboflow/supervision) and [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)
"""
IMAGE_EXAMPLES = [
    [
        os.path.join(os.path.dirname(__file__), "images/livingroom.jpg"),
        "table, lamp, dog, sofa, plant, clock, carpet, frame on the wall",
        0.05,
        0.5,
        True,
        # True,
        True,
    ],
    [
        os.path.join(os.path.dirname(__file__), "images/cat_and_dogs.jpg"),
        "cat, dog",
        0.2,
        0.5,
        True,
        # True,
        True,
    ],
]


# Load models
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

# interence：The confidence score values in the new version of YOLO-World are abnormal due to a bug
# old version not support=============================
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/s")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/m")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/x")

# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-s")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-m")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-l")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-x")
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = EfficientViTSamPredictor(
    create_sam_model(name="xl1", weight_url="./weights/xl1.pt").to(device).eval()
)


# Load annotators
BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(",")]


def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = True,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


def process_image(
    input_image: np.ndarray,
    categories: str,
    confidence_threshold: float,
    nms_threshold: float,
    with_confidence: bool = True,
    # with_class_agnostic_nms: bool = True,
    with_segmentation: bool = True,
) -> np.ndarray:
    global exclude_positions

    # Preparation.
    categories = process_categories(categories)
    YOLO_WORLD_MODEL.set_classes(categories)
    # print("categories:", categories)

    # Object detection
    results = YOLO_WORLD_MODEL.infer(input_image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )
    # print("detected:", detections)

    # Segmentation
    if with_segmentation:
        sam.set_image(input_image, image_format="RGB")
        masks = []
        for xyxy in detections.xyxy:
            mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
            masks.append(mask.squeeze())
        detections.mask = np.array(masks)
        # print("masks shaped as", detections.mask.shape)

    # Annotation
    output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence,
    )

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.005,
    step=0.01,
    label="Confidence Threshold",
    info=(
        "The confidence threshold for the YOLO-World model. Lower the threshold to "
        "reduce false negatives, enhancing the model's sensitivity to detect "
        "sought-after objects. Conversely, increase the threshold to minimize false "
        "positives, preventing the model from identifying objects it shouldn't."
    ),
)

iou_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.5,
    step=0.01,
    label="IoU Threshold",
    info=(
        "The Intersection over Union (IoU) threshold for non-maximum suppression. "
        "Decrease the value to lessen the occurrence of overlapping bounding boxes, "
        "making the detection process stricter. On the other hand, increase the value "
        "to allow more overlapping bounding boxes, accommodating a broader range of "
        "detections."
    ),
)

with_confidence_component = gr.Checkbox(
    value=True,
    label="Display Confidence",
    info=("Whether to display the confidence of the detected objects."),
)

# with_class_agnostic_nms_component = gr.Checkbox(
#     value=True,
#     label="Use Class-Agnostic NMS",
#     info=("Suppress overlapping detections across different classes."),
# )

with_segmentation_component = gr.Checkbox(
    value=True,
    label="With Segmentation",
    info=("Whether to run EfficientViT-SAM for instance segmentation."),
)

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        input_image_component = gr.Image(type="numpy", label="Input Image")
        yolo_world_output_image_component = gr.Image(type="numpy", label="Output image")
    with gr.Row():
        image_categories_text_component = gr.Textbox(
            label="Categories",
            placeholder="you can input multiple words with comma (,)",
            scale=7,
        )
        submit_button_component = gr.Button(value="Submit", scale=1, variant="primary")

    with gr.Accordion("Configuration", open=False):
        confidence_threshold_component.render()
        iou_threshold_component.render()
        with gr.Row():
            with_confidence_component.render()
            # with_class_agnostic_nms_component.render()
            with_segmentation_component.render()

    gr.Examples(
        # fn=process_image,
        examples=IMAGE_EXAMPLES,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_confidence_component,
            # with_class_agnostic_nms_component,
            with_segmentation_component,
        ],
        outputs=yolo_world_output_image_component,
    )

    submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_confidence_component,
            # with_class_agnostic_nms_component,
            with_segmentation_component,
        ],
        outputs=yolo_world_output_image_component,
    )

demo.launch(debug=False, show_error=True)
