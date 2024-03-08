import os
import torch
import comfy
import yaml
import folder_paths

from comfy import model_base, model_management, model_detection, latent_formats, model_sampling
from comfy.supported_models import SD15
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from .configs.modelscope_config import MODELSCOPE_UNET_CONFIG, SD15_CONFIG
from .utils.module_utils import inject_temporal_modules, TEMPORAL_TRANSFORMER_KEY_MAP

MODEL_DIR= "text2video"
model_dir_path = os.path.join(folder_paths.models_dir, MODEL_DIR)

def load_modelscope_checkpoint(
        sd_dict, 
        modelscope_dict, 
        ckpt_path, 
        output_model=True, 
        enable_attn=True, 
        enable_conv = True,
        temporal_attn_strength = 1.0,
        temporal_conv_strength = 1.0,
        sd_15_model=None
    ):
    sd = sd_dict
    modelscope_dict = modelscope_dict
    sd_keys = sd.keys()
    model_patcher = None
    model_scope_config = MODELSCOPE_UNET_CONFIG
    modelscope_dict_process = None

    if sd_15_model is not None:
        model_scope_config = SD15_CONFIG
        if enable_attn:
            print("Modelscope Temporal Attention cannot be enabled when loading a model. Disabling...")
            enable_attn = False
            
        modelscope_dict_process = modelscope_dict.copy()

        for k in modelscope_dict_process.keys():
            if any([key_map in k for key_map in TEMPORAL_TRANSFORMER_KEY_MAP]):
                modelscope_dict.pop(k)

        if all([not enable_conv, not enable_attn]):
            modelscope_dict = {}

    parameters = comfy.utils.calculate_parameters(sd, "model.diffusion_model.")
    load_device = model_management.get_torch_device()

    num_head_channels = MODELSCOPE_UNET_CONFIG['num_head_channels']
    model_config = SD15(model_scope_config)

    if enable_attn:
        model_config.unet_config['num_head_channels'] = num_head_channels

    unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    device = model_management.intermediate_device()

    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = model_management.unet_offload_device()

        model = model_base.BaseModel(model_config, model_type=model_base.ModelType.EPS, device=inital_load_device)
        inject_temporal_modules(
            model, 
            model_scope_config, 
            modelscope_dict,  
            unet_dtype, 
            model_management.intermediate_device(),
            enable_attn=enable_attn,
            enable_conv=enable_conv,
            temporal_attn_strength=temporal_attn_strength,
            temporal_conv_strength=temporal_conv_strength
        )   
        
        # We need to remove the leftover keys if we're using conv processing.
        if all([modelscope_dict_process is not None, enable_conv]):
            for k, v in modelscope_dict_process.items():
                if 'middle_block.3.' in k:
                    del modelscope_dict[k]
                    k = k.replace("middle_block.3.", "middle_block.2.")
                    modelscope_dict[k] = v

                if 'output_blocks.5.3' in k or 'output_blocks.8.3' in k:
                    del modelscope_dict[k]

        sd.update(modelscope_dict)
        model.load_model_weights(sd, "model.diffusion_model.")

        del modelscope_dict_process

    left_over = sd.keys()
    if len(left_over) > 0:
        print("left over keys:", left_over)

    model_patcher = comfy.model_patcher.ModelPatcher(
        model, 
        load_device=load_device, 
        offload_device=model_management.unet_offload_device(), 
        current_device=inital_load_device
    )

    # TODO
    def update_cfg(*args, **kwargs):
        model = kwargs.get("model")
        forward = args[0]
        model_input = args[1]
        x, t, c, c_or_uc = model_input['input'], model_input['timestep'], model_input['c'], model_input['cond_or_uncond']
        is_cfg = len(c_or_uc) > 1
        return forward(x, t, **c)

    if inital_load_device != torch.device("cpu"):
        print("loaded straight to GPU")
        model_management.load_model_gpu(model_patcher)

    return model_patcher

def get_sd_15_keys():
    sd_15_base = UNetModel(**SD15_CONFIG).cpu()
    sd15_keys = sd_15_base.state_dict().keys()
    del sd_15_base

    return sd15_keys

def set_prefix(k):
    if "model.diffusion_model." not in k:
        k = f"model.diffusion_model.{k}"
    return k

def maybe_existing_weight(input_dict: dict, k, v):
    if input_dict is None:
        return v
    try:
        return input_dict[k]
    except:
        print("Failed", k)
        return v

def check_and_apply_keys(state_dict: dict, sd_15_model=None):
    out_dict = {}
    temporal_dict = {}

    sd_15_keys = get_sd_15_keys()

    loaded_dict = (
        sd_15_model.model.diffusion_model.state_dict() \
            if sd_15_model is not None else None
    )
    
    for k, v in state_dict.items():
        # Handle SD 1.5 Base Model
        if k in sd_15_keys or "op." in k:
            if any(['proj_in' in k, 'proj_out' in k]) and len(v.shape) == 2 and loaded_dict is None:
                v = maybe_existing_weight(loaded_dict, k, v)[:, :, None, None]
            
            if ".op." in k:
                k = k.replace(".op", ".0.op")
            out_dict[set_prefix(k)] = maybe_existing_weight(loaded_dict, k, v)

        # Create separate dict for temporal modules and parameters.
        if k not in sd_15_keys:
            temporal_dict[set_prefix(k)] = v
    
    if loaded_dict is not None:
        out_dict = {set_prefix(k): v for k, v in loaded_dict.items()}

    print(f"Spatial Keys (SD): {len(out_dict.keys())}")
    print(f"Total Possible Spatial Keys (SD): {len(sd_15_keys)}")
    print(f"Temporal Keys (ModelScope): {len(temporal_dict.keys())}")

    return out_dict, temporal_dict

class ModelScopeT2VLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (os.listdir(model_dir_path), ),
                "enable_attn": ("BOOLEAN", {"default": True}, ),
                "enable_conv": ("BOOLEAN", {"default": True}, ),
                "temporal_attn_strength": ("FLOAT", {"default": 1.0, "min": 0., "max": 1., "step": 0.1}, ),
                "temporal_conv_strength": ("FLOAT", {"default": 1.0, "min": 0., "max": 1., "step": 0.1}, ),
            },
            "optional": {
                "sd_15_model": ("MODEL", )
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_modelscopet2v"

    def load_modelscopet2v(
        self, 
        model_path, 
        enable_attn, 
        enable_conv, 
        temporal_attn_strength=1.0,
        temporal_conv_strength=1.0,
        sd_15_model=None
    ):
        model_path = os.path.join(model_dir_path, model_path)
        
        if os.path.exists(model_path):
            load_modelscope_dict = comfy.utils.load_torch_file(model_path)
            sd_dict, modelscope_state_dict = check_and_apply_keys(load_modelscope_dict, sd_15_model)
            model = load_modelscope_checkpoint(
                sd_dict, 
                modelscope_state_dict, 
                ckpt_path=None, 
                output_model=True,
                enable_attn=enable_attn,
                enable_conv=enable_conv,
                temporal_attn_strength=temporal_attn_strength,
                temporal_conv_strength=temporal_conv_strength,
                sd_15_model=sd_15_model
            )

        return (model, )

NODE_CLASS_MAPPINGS = {
    "ModelScopeT2VLoader": ModelScopeT2VLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeT2VLoader": "ModelScopeT2VLoader",    
}