from pathlib import Path
import sys
import importlib.util

__version__ = "1.0.0"

current_dir = Path(__file__).parent

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def load_nodes():
    for file in current_dir.glob("AILab_*.py"):
        if file.stem == "__init__":
            continue
        try:
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        except Exception as e:
            print(f"[YoutuVL] Error loading {file.stem}: {e}")


load_nodes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f'\033[34m[ComfyUI-Youtu-VL]\033[0m v\033[93m{__version__}\033[0m | '
      f'\033[93m{len(NODE_CLASS_MAPPINGS)} nodes\033[0m \033[92mLoaded\033[0m')
