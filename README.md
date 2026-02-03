# ComfyUI-Youtu-VL

ComfyUI custom nodes for [Tencent Youtu-VL](https://huggingface.co/tencent/Youtu-VL-4B-Instruct) vision-language model.

Youtu-VL is a lightweight yet powerful 4B parameter VLM with comprehensive vision-centric capabilities including visual grounding, segmentation, depth estimation, and pose estimation.

## Features

| Node | Description |
|------|-------------|
| **YoutuVL** | Basic VLM chat for image understanding |
| **YoutuVL (Advanced)** | Full parameter control for generation |
| **YoutuVL (GGUF)** | GGUF quantized model for lower VRAM |
| **YoutuVL (GGUF Advanced)** | GGUF with full parameter control |
| **YoutuVL Segmentation** | Semantic/referring segmentation with mask output |
| **YoutuVL Grounding** | Visual grounding with bounding boxes |
| **YoutuVL Detection** | Object detection with labels |
| **YoutuVL Depth** | Depth estimation with colormap visualization |
| **YoutuVL Pose** | Human pose estimation with COCO keypoints |


## Installation

### ComfyUI Manager (Recommended)

Search for `ComfyUI-Youtu-VL` in ComfyUI Manager and install.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-Youtu-VL.git
cd ComfyUI-Youtu-VL
pip install -r requirements.txt
```

## Models

Models are automatically downloaded on first use to `ComfyUI/models/LLM/Youtu-VL/`.

| Model | Size | VRAM (FP16) | VRAM (8-bit) | VRAM (4-bit) |
|-------|------|-------------|--------------|--------------|
| Youtu-VL-4B-Instruct | 4B | ~8 GB | ~5 GB | ~3 GB |

### GGUF Support

GGUF quantized models are also available:
- `Youtu-VL-4B-Instruct-GGUF-Q4` - 4-bit quantized
- `Youtu-VL-4B-Instruct-GGUF-Q8` - 8-bit quantized

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.56.0, <= 4.57.1
- CUDA GPU recommended (8GB+ VRAM for FP16)

## Usage

### Basic Chat

1. Add **YoutuVL** node
2. Connect an image
3. Select a preset prompt or enter custom prompt
4. Execute to get text response

### Segmentation

1. Add **YoutuVL Segmentation** node
2. Connect an image
3. Enter prompt describing the object to segment
4. Get segmented image and mask

### Object Detection

1. Add **YoutuVL Detection** or **YoutuVL Grounding** node
2. Connect an image
3. Get annotated image with bounding boxes

## License

- **Code**: GPL-3.0
- **Youtu-VL Model**: Apache-2.0

## Credits

- [Tencent Youtu Lab](https://github.com/TencentCloudADP/youtu-vl) - Youtu-VL model
- [1038lab](https://github.com/1038lab) - ComfyUI integration

## Links

- [Youtu-VL Model](https://huggingface.co/tencent/Youtu-VL-4B-Instruct)
- [Youtu-VL GGUF](https://huggingface.co/tencent/Youtu-VL-4B-Instruct-GGUF)
- [Technical Report](https://arxiv.org/abs/2601.19798)
