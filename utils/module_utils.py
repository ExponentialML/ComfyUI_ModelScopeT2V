import torch.nn as nn
import comfy.ops
import inspect
import copy
from ..modules.modelscope_modules import ResBlock, TemporalTransformer
from comfy.ldm.modules.diffusionmodules.openaimodel import Upsample, Downsample, SpatialTransformer, TimestepEmbedSequential

operations = comfy.ops.disable_weight_init

TEMPORAL_TRANSFORMER_KEY_MAP = ["proj_in", "proj_out", "transformer_blocks", "norm"]
TEMPORAL_RESNET_KEY_MAP = ["temopral_conv"]

def remove_param_str(k):
    return k.replace(".weight", "").replace(".bias", "")

def get_keys_from_dict(input_dict: dict, key_map: list):
    out = []
    for name in input_dict.keys():
        if any([k in name for k in key_map]) and "weight" in name:
            out.append(remove_param_str(name))
    return out

def strip_module_name(keys: list, key_map: list):
    processed_full_keys = []
    out_module_names = []

    for full_key in keys:
        module_name = None
        for key_name in key_map:
            if key_name in full_key:
                module_name = key_name
                break
        
        if module_name is not None:
            if full_key not in processed_full_keys:
                processed_full_keys.append(full_key)
                full_key_split = full_key.split(".")
                stop_idx = full_key_split.index(module_name)
                
                out_full_key = ".".join(full_key_split[:stop_idx])
                out_module_names.append(out_full_key)

    # Remove duplicates
    processed_module_names = []
    for module_name in out_module_names:
        if module_name not in processed_module_names:
            processed_module_names.append(module_name)

    return processed_module_names

def get_submodule_paths(key:str, prefix: str = "model.diffusion_model."):
    # Remove the full path if we have it set.
    if prefix:
        key = key.replace(prefix, "")

    key = key.split(".")
    parent, k = key[:len(key) - 1], key[-1]

    module_path = ".".join(parent)
    module_index = int(k)

    return module_path, module_index

def add_init_temporal_attention(model, unet_config: dict, temporal_attn_strength=1.0):
    model.input_blocks[0].insert(1, 
        TemporalTransformer(
            unet_config['model_channels'], 
            unet_config['num_heads'], 
            unet_config['num_head_channels'],
            context_dim=unet_config['context_dim'],
            depth=1,
            temporal_attn_strength=temporal_attn_strength
        )
    )


def add_temporal_attention_blocks(model, modelscope_dict: dict, temporal_attn_strength=1.0):
    processed_temporal_keys = get_keys_from_dict(modelscope_dict, TEMPORAL_TRANSFORMER_KEY_MAP)
    temporal_transformer_keys = strip_module_name(processed_temporal_keys, TEMPORAL_TRANSFORMER_KEY_MAP)

    for key in temporal_transformer_keys:

        module_path, module_index = get_submodule_paths(key)

        # Get previous module, which is a SpatialTransformer (idx - 1).
        # The current module_index is where the new Temporal Transformer should be.
        module = model.get_submodule(module_path)[module_index - 1]

        if not isinstance(module, SpatialTransformer):
            continue

        n_heads = module.transformer_blocks[0].n_heads
        d_head = module.transformer_blocks[0].d_head

        temporal_transformer_block = TemporalTransformer(
                    module.in_channels, 
                    n_heads, 
                    d_head,
                    depth=1,
                    temporal_attn_strength=temporal_attn_strength
        )
        
        model.get_submodule(module_path).insert(module_index, temporal_transformer_block)
        #print("Verify", model.get_submodule(module_path)[module_index])
        

def replace_proj_resblock(model, modelscope_dict: dict, dtype, device, temporal_conv_strength=1.0):
    processed_resnet_keys = get_keys_from_dict(modelscope_dict, TEMPORAL_RESNET_KEY_MAP)
    temporal_resnet_keys = strip_module_name(processed_resnet_keys, TEMPORAL_RESNET_KEY_MAP)
    
    for key in temporal_resnet_keys:
        
        module_path, module_index = get_submodule_paths(key)

        # Since the temporal module is within the ResBlock, we can just access / assign it directly.
        if module_index == 3:   
            module_index -= 1

        submodule = model.get_submodule(module_path)[module_index]

        res_block = ResBlock(
                channels=submodule.channels,
                emb_channels=submodule.emb_channels,
                dropout=0.0,
                out_channels=submodule.out_channels,
                use_scale_shift_norm=submodule.use_scale_shift_norm,
                down=isinstance(submodule.h_upd, Downsample),
                up=isinstance(submodule.h_upd, Upsample),
                dtype=dtype,
                device=device,
                operations=operations,
                use_temporal_conv=True,
                temporal_conv_strength=temporal_conv_strength
            )
            
        model.get_submodule(module_path)[module_index] = res_block
        #print("Verify", model.get_submodule(module_path)[module_index])

def inject_temporal_modules(
    model, 
    unet_config, 
    modelscope_dict, 
    dtype, 
    device, 
    enable_attn=True, 
    enable_conv=True,
    temporal_attn_strength=1.0,
    temporal_conv_strength=1.0
):
    if enable_conv:
        replace_proj_resblock(model.diffusion_model, modelscope_dict, dtype, device, temporal_conv_strength)
    if enable_attn:
        add_init_temporal_attention(model.diffusion_model, unet_config, temporal_attn_strength)      
        add_temporal_attention_blocks(model.diffusion_model, modelscope_dict, temporal_attn_strength)
        