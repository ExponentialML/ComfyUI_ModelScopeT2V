# ComfyUI_ModelScopeT2V
![image](https://github.com/ExponentialML/ComfyUI_ModelScopeT2V/assets/59846140/724b8150-eb30-4f1f-9f3f-c3dc17233825)


Allows native usage of ModelScope based Text To Video Models in ComfyUI

## Getting Started

### Clone The Repository
```bash
cd /your/path/to/ComfyUI/custom_nodes
git clone https://github.com/ExponentialML/ComfyUI_ModelScopeT2V.git
```

### Preparation
Create a folder in your ComfyUI `models` folder named `text2video`.

## Download Models
Models that were converted to A1111 format will work. 

### Modelscope
https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/tree/main

### Zeroscope
https://huggingface.co/cerspense/zeroscope_v2_1111models

### Instructions
Place the models in `text2video_pytorch_model.pth` model in the `text2video` directory.

You must also use the accompanying `open_clip_pytorch_model.bin`, and place it in the `clip` folder under your `model` directory.

This is optional if you're not using the attention layers, and are using something like AnimateDiff (more on this in usage).

## Usage

- `model_path`: The path to your ModelScope model.

- `enable_attn`: Enables the temporal attention of the ModelScope model. If this is disabled, you must apply a 1.5 based model. If this option is enabled and you apply a 1.5 based model, this parameter will be disabled by default. This is due to ModelScope's usage of the SD 2.0 based CLIP model instead of the 1.5 one.

- `enable_conv`: Enables the temporal convolution modules of the ModelScope model. Enabling this option with a 1.5 based model as input will allow you to leverage temporal convoutions with other modules (such as AnimateDiff)

- `temporal_attn_strength`: Controls the strength of the temporal attention, bringing it closer to the dataset input without temporal properties.

- `temporal_conv_strength`: Controls the strength of the temporal convolution, bringing it closer to the model input without temporal properties.

- `sd_15_model`: Optional. If left blank, pure ModelScope will be used.

### Tips
1. If you're using pure ModelScope, try higher CFG (around 15) for better coherence. You may also try any other rescale nodes.
2. If using pure ModelScope, ensure that you use a minimum of 24 frames.
3. If using with AnimateDiff, make sure to use 16 frames if you're not using context options.


## TODO
- [ ] Uncoditional guidance (CFG 1) is currently not implemented.
- [ ] Explore ensembling 1.5 models with the 2.0 CLIP encoder to use all modules.

## Atributions
The temporal code was borrowed and leveraged from https://github.com/kabachuha/sd-webui-text2video. Thanks @kabachuha!
Thanks to the ModelScope team for open sourcing. Check out there [existing works](https://github.com/modelscope)https://github.com/modelscope.
