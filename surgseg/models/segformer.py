from transformers import SegformerForSemanticSegmentation
from ..config import CLASS_NAMES

def create_segformer(num_classes, variant="b2", ignore_index=255):
    ckpt_map = {
        "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }
    name = ckpt_map.get(variant, ckpt_map["b2"])
    model = SegformerForSemanticSegmentation.from_pretrained(
        name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    id2label = {i: cls for i, cls in enumerate(CLASS_NAMES)}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.semantic_loss_ignore_index = ignore_index
    return model
