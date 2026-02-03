# ComfyUI-Youtu-VL
# Custom nodes for Tencent Youtu-VL vision-language model
# Model: https://huggingface.co/tencent/Youtu-VL-4B-Instruct
# Code License: GPL-3.0
# Model License: License Term of Youtu-VL (Non-Commercial, See HF for details)
# Source: https://github.com/1038lab/ComfyUI-Youtu-VL

import gc
import json
import sys
import types
from enum import Enum
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

import folder_paths

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}

TOOLTIPS = {
    "model_name": "Select the Youtu-VL model. First run downloads weights to models/LLM/Youtu-VL.",
    "quantization": "Precision vs VRAM. FP16 gives best quality; 8-bit suits 8-16GB GPUs; 4-bit fits 6GB or less.",
    "attention_mode": "auto tries flash-attn v2 when available, falls back to SDPA.",
    "preset_prompt": "Built-in instruction for how Youtu-VL should analyze the input.",
    "custom_prompt": "Optional override - replaces preset template when filled.",
    "max_tokens": "Maximum number of new tokens to generate.",
    "keep_model_loaded": "Keep model in VRAM after run for faster subsequent inference.",
    "seed": "Seed for reproducible results.",
    "temperature": "Sampling randomness. 0.1-0.4 is focused, 0.7+ is creative.",
    "top_p": "Nucleus sampling cutoff. Lower values keep only top tokens.",
    "repetition_penalty": "Values >1 penalize repeated phrases.",
}


class Quantization(str, Enum):
    Q4 = "4-bit (VRAM-friendly)"
    Q8 = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Unsupported quantization: {value}")


ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa"]


def load_model_configs():
    global MODEL_CONFIGS, SYSTEM_PROMPTS
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            MODEL_CONFIGS = json.load(fh)
            SYSTEM_PROMPTS = MODEL_CONFIGS.get("_system_prompts", {})
    except Exception as exc:
        print(f"[YoutuVL] Config load failed: {exc}")
        MODEL_CONFIGS = {}
        SYSTEM_PROMPTS = {}


if not MODEL_CONFIGS:
    load_model_configs()


def get_device_info():
    gpu = {"available": False, "total_memory": 0, "free_memory": 0}
    device_type = "cpu"
    recommended = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        gpu = {
            "available": True,
            "total_memory": total,
            "free_memory": total - (torch.cuda.memory_allocated(0) / 1024**3),
        }
        device_type = "nvidia_gpu"
        recommended = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
        gpu = {"available": True, "total_memory": 0, "free_memory": 0}
    sys_mem = psutil.virtual_memory()
    return {
        "gpu": gpu,
        "system_memory": {
            "total": sys_mem.total / 1024**3,
            "available": sys_mem.available / 1024**3,
        },
        "device_type": device_type,
        "recommended_device": recommended,
    }


def flash_attn_available():
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_attention_mode(mode):
    if mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[YoutuVL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"
    if flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def ensure_model(model_name):
    info = MODEL_CONFIGS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]
    models_dir = Path(folder_paths.models_dir) / "LLM" / "Youtu-VL"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".git*"],
    )
    return str(target)


def enforce_memory(model_name, quantization, device_info):
    info = MODEL_CONFIGS.get(model_name, {})
    requirements = info.get("vram_requirement", {})
    mapping = {
        Quantization.Q4: requirements.get("4bit", 0),
        Quantization.Q8: requirements.get("8bit", 0),
        Quantization.FP16: requirements.get("full", 0),
    }
    needed = mapping.get(quantization, 0)
    if not needed:
        return quantization
    if device_info["recommended_device"] in {"cpu", "mps"}:
        needed *= 1.5
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]
    if needed * 1.2 > available:
        if quantization == Quantization.FP16:
            print("[YoutuVL] Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[YoutuVL] Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
        raise RuntimeError("Insufficient memory for 4-bit mode")
    return quantization


def quantization_config(model_name, quantization):
    info = MODEL_CONFIGS.get(model_name, {})
    if info.get("quantized"):
        return None, None

    # Skip vision encoder modules during quantization to avoid LayerNorm errors
    # Youtu-VL uses SigLIP2, usually named 'siglip2' or 'vision_model'
    if quantization == Quantization.Q4:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return cfg, None
    if quantization == Quantization.Q8:
        # Skip vision encoder modules during 8-bit quantization to avoid LayerNorm Int8 errors
        # Youtu-VL uses SigLIP2, usually named 'siglip2' or 'vision_model'
        # Also skip lm_head to avoid 'lm_head.SCB' bitsandbytes loading error
        skip_modules = ["siglip2", "vision_model", "vision_tower", "image_tower", "lm_head"]
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=skip_modules,
        ), None
    return None, torch.float16 if torch.cuda.is_available() else torch.float32


class YoutuVLBase:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.current_signature = None
        print(f"[YoutuVL] Node on {self.device_info['device_type']}")

    def clear(self):
        self.model = None
        self.processor = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        device_choice,
        keep_model_loaded,
    ):
        quant = enforce_memory(model_name, Quantization.from_value(quant_value), self.device_info)
        attn_impl = resolve_attention_mode(attention_mode)
        device = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        signature = (model_name, quant.value, attn_impl, device)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            print(f"[YoutuVL] Reusing loaded model: {model_name} ({quant.value})")
            return
        self.clear()
        model_path = ensure_model(model_name)
        quant_config, dtype = quantization_config(model_name, quant)
        load_kwargs = {
            "device_map": {"": 0} if device == "cuda" and torch.cuda.is_available() else device,
            "torch_dtype": dtype if dtype else "auto",
            "attn_implementation": attn_impl,
            "trust_remote_code": True,
        }
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        print(f"[YoutuVL] Loading {model_name} ({quant.value}, attn={attn_impl})")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()
        self.model.config.use_cache = True
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.current_signature = signature

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        image,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
    ):
        messages = [{"role": "user", "content": []}]
        img_path = None
        if image is not None:
            pil_img = self.tensor_to_pil(image)
            img_path = Path(folder_paths.get_temp_directory()) / "youtu_vl_input.png"
            pil_img.save(str(img_path))
            messages[0]["content"].append({"type": "image", "image": str(img_path)})
        messages[0]["content"].append({"type": "text", "text": prompt_text})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
        }
        if img_path:
            gen_kwargs["img_input"] = str(img_path)

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0].strip()

    def run(
        self,
        model_name,
        quantization,
        preset_prompt,
        custom_prompt,
        image,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        seed,
        keep_model_loaded,
        attention_mode,
        device,
    ):
        torch.manual_seed(seed)
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        self.load_model(model_name, quantization, attention_mode, device, keep_model_loaded)
        try:
            text = self.generate(prompt, image, max_tokens, temperature, top_p, repetition_penalty)
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_YoutuVL(YoutuVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = [name for name in MODEL_CONFIGS.keys() if not name.startswith("_") and not MODEL_CONFIGS.get(name, {}).get("is_gguf")]
        default_model = next((m for m in models if MODEL_CONFIGS.get(m, {}).get("default")), models[0] if models else "Youtu-VL-4B-Instruct")
        prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image."])
        default_prompt = "🖼️ Describe Image" if "🖼️ Describe Image" in prompts else prompts[0]
        return {
            "required": {
                "model": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
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
        quantization,
        attention_mode,
        preset_prompt,
        custom_prompt,
        max_tokens,
        keep_model_loaded,
        seed,
        image=None,
    ):
        try:
            return self.run(
                model,
                quantization,
                preset_prompt,
                custom_prompt,
                image,
                max_tokens,
                0.1,
                0.001,
                1.05,
                seed,
                keep_model_loaded,
                attention_mode,
                "auto",
            )
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_YoutuVL_Advanced(YoutuVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = [name for name in MODEL_CONFIGS.keys() if not name.startswith("_") and not MODEL_CONFIGS.get(name, {}).get("is_gguf")]
        default_model = next((m for m in models if MODEL_CONFIGS.get(m, {}).get("default")), models[0] if models else "Youtu-VL-4B-Instruct")
        prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image."])
        default_prompt = "🖼️ Describe Image" if "🖼️ Describe Image" in prompts else prompts[0]
        return {
            "required": {
                "model": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 32768, "tooltip": TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": TOOLTIPS["temperature"]}),
                "top_p": ("FLOAT", {"default": 0.001, "min": 0.001, "max": 1.0, "step": 0.001, "tooltip": TOOLTIPS["top_p"]}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": TOOLTIPS["repetition_penalty"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
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
        quantization,
        attention_mode,
        device,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        image=None,
    ):
        try:
            return self.run(
                model,
                quantization,
                preset_prompt,
                custom_prompt,
                image,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
                seed,
                keep_model_loaded,
                attention_mode,
                device,
            )
        finally:
            if not keep_model_loaded:
                self.clear()


NODE_CLASS_MAPPINGS = {
    "AILab_YoutuVL": AILab_YoutuVL,
    "AILab_YoutuVL_Advanced": AILab_YoutuVL_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_YoutuVL": "Youtu-VL",
    "AILab_YoutuVL_Advanced": "Youtu-VL (Advanced)",
}
