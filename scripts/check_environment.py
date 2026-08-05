from __future__ import annotations

import importlib
import json
import platform
import sys

REQUIRED = ("torch", "numpy", "scipy", "yaml", "PIL", "tqdm")


def main() -> int:
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    missing = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            result["packages"][name] = getattr(module, "__version__", "installed")
        except ImportError:
            result["packages"][name] = None
            missing.append(name)
    if "torch" not in missing:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
        result["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(json.dumps(result, indent=2))
    if missing:
        print(f"Missing required packages: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
