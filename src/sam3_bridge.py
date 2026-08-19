"""Drive SAM3's detector from C-RADIOv4 features instead of SAM3's own backbone.

C-RADIOv4 distils `sam3` as one of its three teachers and the release ships
`_feature_projections.sam3`, an MLP taking its 1152-wide patch tokens to the
1024-wide space SAM3's ViT produces. Everything above that point -- the FPN neck,
the DETR encoder/decoder, the mask decoder -- is SAM3's and is reused unchanged.
If the projection is faithful, one C-RADIOv4 pass then yields text-promptable
detection and segmentation, and the separate detector comes out of the pipeline.

Two details make the wiring exact rather than approximate.

`Sam3Model.forward` takes a `vision_embeds` argument, so the vision encoder can be
bypassed outright; nothing has to be monkey-patched and the rest of the model runs
on its own code path.

The grids have to be made to agree. SAM3's ViT is patch 14 at 1008px, giving
72x72, while C-RADIOv4 is patch 16 -- so it is run at **1152px**, which is 72x16
and lands on the same 72x72 grid the neck and the DETR position encodings were
trained against. Feeding a different grid would work mechanically, since the neck
is convolutional, but would put every spatial prior off scale.

Always compare against `native=True`, which runs SAM3's own backbone on the same
images. Without that control a poor number cannot be attributed: it could be the
projection failing, or SAM3 simply not working on aerial imagery.
"""
from __future__ import annotations

import torch

RADIO_SIZE = 1152          # 72 * 16, matching SAM3's 72x72 grid at patch 14 / 1008
GRID = 72


def load_sam3(model_id: str = "facebook/sam3", device: str = "cuda",
              dtype: torch.dtype = torch.float32):
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device).eval()
    proc = Sam3Processor.from_pretrained(model_id)
    return model, proc


def radio_vision_embeds(sam3, patches: torch.Tensor, projection):
    """C-RADIOv4 patch tokens -> the Sam3VisionEncoderOutput the detector expects.

    `patches` is [B, GRID*GRID, 1152] from a 1152px C-RADIOv4 pass.
    """
    from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

    B, N, _ = patches.shape
    if N != GRID * GRID:
        raise ValueError(
            f"{N} patch tokens; SAM3's neck wants {GRID*GRID} ({GRID}x{GRID}). "
            f"Run C-RADIOv4 at {RADIO_SIZE}px.")

    p = next(projection.parameters())
    x = projection(patches.to(p.dtype))                       # [B, N, 1024]
    neck_in = x.view(B, GRID, GRID, -1).permute(0, 3, 1, 2)   # [B, 1024, 72, 72]
    neck = sam3.vision_encoder.neck
    nd = next(neck.parameters()).dtype
    fpn, pos = neck(neck_in.to(nd))
    return Sam3VisionEncoderOutput(
        last_hidden_state=x.to(nd), fpn_hidden_states=fpn, fpn_position_encoding=pos)


@torch.no_grad()
def detect(sam3, proc, images, text, vision_embeds=None, device="cuda",
           threshold: float = 0.3):
    """Text-prompted detection. `vision_embeds=None` runs SAM3's own backbone.

    -> list per image of {'boxes': [[x0,y0,x1,y1]], 'scores': [...], 'masks': ...}
    """
    enc = proc(images=images, text=[text] * len(images), return_tensors="pt")
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    if vision_embeds is not None:
        enc.pop("pixel_values", None)
        out = sam3(vision_embeds=vision_embeds, input_ids=enc["input_ids"],
                   attention_mask=enc.get("attention_mask"))
    else:
        out = sam3(**enc)
    sizes = [(im.height, im.width) for im in images]
    return proc.post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=0.5, target_sizes=sizes)
