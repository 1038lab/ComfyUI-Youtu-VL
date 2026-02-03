# ComfyUI-Youtu-VL GGUF Node
# Custom nodes for Tencent Youtu-VL vision-language model
# Model: https://huggingface.co/tencent/Youtu-VL-4B-Instruct
# Code License: GPL-3.0
# Model License: License Term of Youtu-VL (Non-Commercial, See HF for details)
# Source: https://github.com/1038lab/ComfyUI-Youtu-VL

import gc
import json
import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

import folder_paths

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}

TOOLTIPS = {
    "model_name": "Select the GGUF quantized model.",
    "n_gpu_layers": "Number of layers to offload to GPU. -1 = all layers.",
    "n_ctx": "Context window size.",
    "max_tokens": "Maximum number of new tokens to generate.",
    "temperature": "Sampling temperature. Lower = more focused.",
    "top_p": "Nucleus sampling cutoff.",
    "keep_model_loaded": "Keep model loaded for faster subsequent inference.",
}


def load_model_configs():
    global MODEL_CONFIGS, SYSTEM_PROMPTS
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            MODEL_CONFIGS = json.load(fh)
            SYSTEM_PROMPTS = MODEL_CONFIGS.get("_system_prompts", {})
    except Exception as exc:
        print(f"[YoutuVL-GGUF] Config load failed: {exc}")
        MODEL_CONFIGS = {}
        SYSTEM_PROMPTS = {}


if not MODEL_CONFIGS:
    load_model_configs()


def get_gguf_models():
    return {k: v for k, v in MODEL_CONFIGS.items() if not k.startswith("_") and v.get("is_gguf")}


def ensure_gguf_model(model_name):
    info = MODEL_CONFIGS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]
    filename = info.get("filename", "Youtu-VL-4B-Instruct-Q8_0.gguf")
    mmproj = info.get("mmproj", "mmproj-Youtu-VL-4b-Instruct-BF16.gguf")
    model_id = repo_id.split("/")[-1]
    models_dir = Path(folder_paths.models_dir) / "LLM" / "GGUF" / model_id
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / filename
    if not model_path.exists():
        print(f"[YoutuVL-GGUF] Downloading {filename} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(models_dir),
        )

    mmproj_path = models_dir / mmproj
    if not mmproj_path.exists():
        print(f"[YoutuVL-GGUF] Downloading {mmproj} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=mmproj,
            local_dir=str(models_dir),
        )

    return str(model_path), str(mmproj_path)


class YoutuVLGGUFBase:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        print("[YoutuVL-GGUF] Node initialized")

    def clear(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.current_model_name = None
        gc.collect()

    def load_model(self, model_name, n_gpu_layers, n_ctx, keep_model_loaded):
        if keep_model_loaded and self.model is not None and self.current_model_name == model_name:
            print(f"[YoutuVL-GGUF] Reusing loaded model: {model_name}")
            return

        self.clear()
        model_path, mmproj_path = ensure_gguf_model(model_name)

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        print(f"[YoutuVL-GGUF] Loading {model_name}...")
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
        self.model = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            use_mmap=False,
            verbose=False,
        )
        self.current_model_name = model_name

    @staticmethod
    def tensor_to_base64(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(array)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(self, prompt_text, image, max_tokens, temperature, top_p):
        messages = []

        if image is not None:
            img_b64 = self.tensor_to_base64(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            })
        else:
            messages.append({"role": "user", "content": prompt_text})

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return response["choices"][0]["message"]["content"].strip()


class AILab_YoutuVL_GGUF(YoutuVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        gguf_models = get_gguf_models()
        models = list(gguf_models.keys()) if gguf_models else ["Youtu-VL-4B-Instruct-GGUF-Q4"]
        prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image."])
        default_prompt = "🖼️ Describe Image" if "🖼️ Describe Image" in prompts else prompts[0]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": TOOLTIPS["model_name"]}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/YoutuVL"

    def process(
        self,
        model,
        preset_prompt,
        custom_prompt,
        max_tokens,
        keep_model_loaded,
        seed,
        image=None,
    ):
        torch.manual_seed(seed)
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        self.load_model(model, -1, 4096, keep_model_loaded)
        try:
            text = self.generate(prompt, image, max_tokens, 0.1, 0.001)
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_YoutuVL_GGUF_Advanced(YoutuVLGGUFBase):
    @classmethod
    def INPUT_TYPES(cls):
        gguf_models = get_gguf_models()
        models = list(gguf_models.keys()) if gguf_models else ["Youtu-VL-4B-Instruct-GGUF-Q4"]
        prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image."])
        default_prompt = "🖼️ Describe Image" if "🖼️ Describe Image" in prompts else prompts[0]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": TOOLTIPS["model_name"]}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100, "tooltip": TOOLTIPS["n_gpu_layers"]}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "tooltip": TOOLTIPS["n_ctx"]}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 32768, "tooltip": TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": TOOLTIPS["temperature"]}),
                "top_p": ("FLOAT", {"default": 0.001, "min": 0.001, "max": 1.0, "step": 0.001, "tooltip": TOOLTIPS["top_p"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/YoutuVL"

    def process(
        self,
        model,
        n_gpu_layers,
        n_ctx,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        keep_model_loaded,
        seed,
        image=None,
    ):
        torch.manual_seed(seed)
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        self.load_model(model, n_gpu_layers, n_ctx, keep_model_loaded)
        try:
            text = self.generate(prompt, image, max_tokens, temperature, top_p)
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


NODE_CLASS_MAPPINGS = {
    "AILab_YoutuVL_GGUF": AILab_YoutuVL_GGUF,
    "AILab_YoutuVL_GGUF_Advanced": AILab_YoutuVL_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_YoutuVL_GGUF": "Youtu-VL (GGUF)",
    "AILab_YoutuVL_GGUF_Advanced": "Youtu-VL (GGUF Advanced)",
}
