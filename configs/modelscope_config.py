import torch
from comfy import model_base
from comfy import utils

from comfy import sd1_clip
from comfy import sd2_clip

from comfy import supported_models_base
from comfy import latent_formats

SD15_CONFIG = {
    'use_checkpoint': False, 
    'image_size': 32, 
    'out_channels': 4, 
    'use_spatial_transformer': True, 
    'legacy': False, 
    'adm_in_channels': None,
    'in_channels': 4, 
    'model_channels': 320, 
    'num_res_blocks': [2, 2, 2, 2], 
    'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
    'channel_mult': [1, 2, 4, 4], 
    'transformer_depth_middle': 1, 
    'use_linear_in_transformer': False, 
    'context_dim': 768, 
    'num_heads': 8,
    'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'use_temporal_attention': False, 
    'use_temporal_resblock': False
}

MODELSCOPE_UNET_CONFIG = {
    "use_spatial_transformer": True,
    "model_channels": 320,
    "in_channels": 4,
    "out_channels": 4,
    "context_dim": 1024,
    'num_res_blocks': [2, 2, 2, 2],
    "channel_mult": (1, 2, 4, 4),
    "num_heads": 8,
    "num_head_channels": 64,
    "image_size": 32,
    "use_linear_in_transformer": False,
    "adm_in_channels": None,
    "use_temporal_attention": False,
    "merge_strategy": "fixed",
    'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
    'transformer_depth_middle': 1,
    'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
}

class SD15ModelScope(supported_models_base.BASE):
    unet_extra_config = {
        "num_heads": MODELSCOPE_UNET_CONFIG['num_heads'],
        "num_head_channels": MODELSCOPE_UNET_CONFIG['num_head_channels'],
    }
    latent_format = latent_formats.SD15

    def model_type(self, state_dict, prefix=""):
        if self.unet_config["in_channels"] == 4: #SD2.0 inpainting models are not v prediction
            k = "{}output_blocks.11.1.transformer_blocks.0.norm1.bias".format(prefix)
            out = state_dict[k]
            if torch.std(out, unbiased=False) > 0.09: # not sure how well this will actually work. I guess we will find out.
                return model_base.ModelType.V_PREDICTION
        return model_base.ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_h." #SD2 in sgm format
        replace_prefix["cond_stage_model.model."] = "clip_h."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        state_dict = utils.clip_text_transformers_convert(state_dict, "clip_h.", "clip_h.transformer.")
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        replace_prefix["clip_h"] = "cond_stage_model.model"
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        state_dict = diffusers_convert.convert_text_enc_state_dict_v20(state_dict)
        return state_dict

    def clip_target(self):
        return supported_models_base.ClipTarget(sd2_clip.SD2Tokenizer, sd2_clip.SD2ClipModel)