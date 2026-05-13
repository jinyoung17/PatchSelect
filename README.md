# PatchSelect

Official PyTorch implementation of PatchSelect for patch-based text-driven image editing.

## Editing Instances

The file `data/editing_instances.json` contains all attribute–object editing instances used in the experiments. Each entry defines a source image (`init_img`) and multiple target editing prompts (`target_prompts`). Each image–prompt pair corresponds to a single editing instance.

## Installation

```bash
pip install -r requirements.txt
```

## Training
```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py \
  --image_file_path final_images/zebra.jpeg \
  --image_caption "zebra" \
  --editing_prompt "blue horse" \
  --diffusion_model_path "stabilityai/stable-diffusion-2-inpainting" \
  --image_size 512 \
  --epochs 5 \
  --pin_mem \
  --lr 5e-3
```

If `stabilityai/stable-diffusion-2-inpainting` is unavailable, use
`sd2-community/stable-diffusion-2-inpainting` instead.
