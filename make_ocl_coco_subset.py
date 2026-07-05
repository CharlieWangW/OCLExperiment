import argparse
import pickle
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a tiny OCL subset from COCO val2014 annotations/images.")
    parser.add_argument("--official-root", default="ObjectConceptLearning")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--coco-zip", default="data/COCO/val2014.zip")
    parser.add_argument("--output", default="ocl_coco_subset")
    parser.add_argument("--top-categories", type=int, default=10)
    parser.add_argument("--per-category", type=int, default=4)
    return parser.parse_args()


def load_json_like(path):
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def map_attribute(label_id, labels):
    if isinstance(label_id, int) and 0 <= label_id < len(labels):
        return str(labels[label_id])
    return str(label_id)


def map_affordance(label_id, labels):
    if isinstance(label_id, int) and 0 <= label_id < len(labels):
        item = labels[label_id]
        if isinstance(item, dict):
            words = item.get("word") or item.get("words")
            if isinstance(words, list) and words:
                return str(words[0])
            if "name" in item:
                return str(item["name"])
        return str(item)
    return str(label_id)


def coco_zip_member(ocl_name):
    # OCL names look like COCO/val2014/COCO_val2014_000000087383.jpg.
    parts = Path(ocl_name).parts
    if len(parts) >= 3 and parts[0] == "COCO":
        return str(Path(*parts[1:])).replace("\\", "/")
    raise ValueError(f"Not a COCO OCL path: {ocl_name}")


def main():
    args = parse_args()
    official_root = Path(args.official_root)
    resource_dir = official_root / "data" / "resources"
    annotation_path = resource_dir / f"OCL_annot_{args.split}.pkl"
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    coco_zip = Path(args.coco_zip)
    if not coco_zip.exists():
        raise FileNotFoundError(coco_zip)

    attribute_labels = load_json_like(resource_dir / "OCL_class_attribute.json")
    affordance_labels = load_json_like(resource_dir / "OCL_class_affordance.json")

    with open(annotation_path, "rb") as handle:
        records = pickle.load(handle)

    category_counts = Counter()
    flattened = []
    for record in records:
        if record.get("source") != "COCO":
            continue
        for obj in record.get("objects", []):
            category = obj.get("obj")
            if not category:
                continue
            category_counts[category] += 1
            flattened.append((record, obj))

    selected_categories = {name for name, _ in category_counts.most_common(args.top_categories)}
    per_category_counts = defaultdict(int)
    selected = []
    for record, obj in flattened:
        category = obj.get("obj")
        if category not in selected_categories:
            continue
        if per_category_counts[category] >= args.per_category:
            continue
        per_category_counts[category] += 1
        selected.append((record, obj))
        if len(per_category_counts) == len(selected_categories) and all(
            per_category_counts[name] >= args.per_category for name in selected_categories
        ):
            break

    output_root = Path(args.output)
    image_dir = output_root / "data" / "images"
    subset_resource_dir = output_root / "data" / "resource"
    image_dir.mkdir(parents=True, exist_ok=True)
    subset_resource_dir.mkdir(parents=True, exist_ok=True)

    subset_records = []
    with zipfile.ZipFile(coco_zip) as archive:
        available = set(archive.namelist())
        for index, (record, obj) in enumerate(selected):
            member = coco_zip_member(record["name"])
            if member not in available:
                print(f"skip missing in zip: {member}")
                continue
            image_name = Path(member).name
            target_image = image_dir / image_name
            if not target_image.exists():
                with archive.open(member) as src, open(target_image, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            attrs = [map_attribute(value, attribute_labels) for value in obj.get("attr", [])]
            affs = [map_affordance(value, affordance_labels) for value in obj.get("aff", [])]
            subset_records.append(
                {
                    "image_path": f"images/{image_name}",
                    "source": "COCO",
                    "objects": [
                        {
                            "category": obj["obj"],
                            "bbox": obj.get("box"),
                            "attributes": attrs,
                            "affordances": affs,
                            "original_index": index,
                        }
                    ],
                }
            )

    output_annotation = subset_resource_dir / "OCL_class_valtest.pkl"
    with open(output_annotation, "wb") as handle:
        pickle.dump(subset_records, handle)

    print(f"selected categories: {sorted(selected_categories)}")
    print(f"written samples: {len(subset_records)}")
    print(f"output root: {output_root.resolve()}")
    print(f"annotation: {output_annotation.resolve()}")


if __name__ == "__main__":
    main()

