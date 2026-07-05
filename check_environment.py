import importlib.util
import sys


REQUIRED = {
    "torch": "torch",
    "torchvision": "torchvision",
    "PIL": "pillow",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
    "clip": "git+https://github.com/openai/CLIP.git",
}


def main():
    missing = []
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    for module_name, package_name in REQUIRED.items():
        ok = importlib.util.find_spec(module_name) is not None
        print(f"{module_name:12s} {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(package_name)

    if missing:
        print()
        print("missing packages:")
        for package_name in missing:
            print(f"  - {package_name}")
        print()
        print("install command:")
        print("  pip install -r requirements.txt")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
