"""Bound the ViTDet attention workspace during 4D-Humans inference."""
import torch


def attention_by_head(module, image, add_relative_position):
    batch, height, width, _ = image.shape
    tokens = height * width
    heads = module.num_heads
    qkv = module.qkv(image).reshape(batch, tokens, 3, heads, -1).permute(2, 0, 3, 1, 4)
    queries, keys, values = qkv.reshape(3, batch * heads, tokens, -1).unbind(0)
    outputs = []
    for query, key, value in zip(queries.split(2), keys.split(2), values.split(2), strict=True):
        weights = (query * module.scale) @ key.transpose(-2, -1)
        if module.use_rel_pos:
            weights = add_relative_position(
                weights, query, module.rel_pos_h, module.rel_pos_w, (height, width), (height, width)
            )
        outputs.append(weights.softmax(dim=-1) @ value)
        del weights
    features = torch.cat(outputs).reshape(batch, heads, height, width, -1).permute(0, 2, 3, 1, 4)
    return module.proj(features.reshape(batch, height, width, -1))


def limit_vitdet_workspace():
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory > 16 * 2**30:
        return
    from detectron2.modeling.backbone import vit

    original_forward = vit.Attention.forward

    def forward(attention, image):
        if attention.training or torch.is_grad_enabled() or not image.is_cuda or image.shape[1] * image.shape[2] < 2048:
            return original_forward(attention, image)
        return attention_by_head(attention, image, vit.add_decomposed_rel_pos)

    vit.Attention.forward = forward
    print("4D-Humans: limiting ViTDet global attention to two heads at a time.", flush=True)
