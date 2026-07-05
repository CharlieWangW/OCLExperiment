import json
import pickle
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _clean_label(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, float)):
        return str(int(value))
    text = str(value).strip()
    return text if text else None


def _first(mapping, keys):
    for key in keys:
        if isinstance(mapping, dict) and key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class OCLDataset(Dataset):
    """Object-level OCL dataset for CLIP inference.

    The official OCL annotations are pickle files containing image-level records.
    Each image record contains several object instances. This loader converts the
    image-level structure into object-level samples and crops each object by its
    bounding box before applying CLIP preprocessing.
    """

    image_keys = (
        "image",
        "img",
        "file_name",
        "filename",
        "image_name",
        "img_name",
        "image_path",
        "img_path",
        "path",
        "file",
        "name",
        "md5",
    )
    object_keys = ("objects", "instances", "annotations", "objs", "regions", "object")
    category_keys = (
        "category",
        "category_name",
        "class",
        "class_name",
        "object_category",
        "object_name",
        "label",
        "name",
    )
    attribute_keys = (
        "attributes",
        "attribute",
        "attrs",
        "attr",
        "positive_attributes",
        "attr_label",
        "attribute_labels",
    )
    affordance_keys = (
        "affordances",
        "affordance",
        "affs",
        "aff",
        "positive_affordances",
        "affordance_label",
        "affordance_labels",
    )
    bbox_keys = ("bbox", "box", "bndbox", "bounding_box", "object_box", "rect")

    def __init__(
        self,
        data_root,
        split="valtest",
        preprocess=None,
        tokenizer=None,
        top_categories=10,
        max_samples=None,
        load_images=True,
    ):
        self.data_root = Path(data_root)
        self.data_dir = self.data_root / "data"
        self.resource_dir = self._find_resource_dir()
        self.annotation_file = self._find_annotation_file(split)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.load_images = load_images
        self._image_index = None

        with self.annotation_file.open("rb") as handle:
            self.raw_annotations = pickle.load(handle)

        self.vocabs = self._load_vocabs()
        all_samples = self._flatten_annotations()
        if not all_samples:
            raise RuntimeError(f"No object samples parsed from {self.annotation_file}")

        category_counts = Counter(sample["category"] for sample in all_samples)
        if top_categories is None:
            self.selected_categories = [name for name, _ in category_counts.most_common()]
        else:
            self.selected_categories = [name for name, _ in category_counts.most_common(top_categories)]
        selected = set(self.selected_categories)
        self.samples = [sample for sample in all_samples if sample["category"] in selected]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.attribute_labels = sorted({label for sample in self.samples for label in sample["attributes"]})
        self.affordance_labels = sorted({label for sample in self.samples for label in sample["affordances"]})
        self.attribute_to_index = {label: idx for idx, label in enumerate(self.attribute_labels)}
        self.affordance_to_index = {label: idx for idx, label in enumerate(self.affordance_labels)}

    def _find_resource_dir(self):
        candidates = [
            self.data_dir / "resource",
            self.data_dir / "resources",
            self.data_root / "resource",
            self.data_root / "resources",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not find OCL resource directory. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def _find_annotation_file(self, split):
        patterns = [
            f"OCL_class_{split}.pkl",
            f"*{split}*.pkl",
            "OCL_class_valtest.pkl",
            "OCL_class_test.pkl",
            "*.pkl",
        ]
        for pattern in patterns:
            matches = sorted(self.resource_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No OCL annotation pickle found in {self.resource_dir}")

    def _load_vocabs(self):
        vocabs = {}
        for path in self.resource_dir.glob("*.json"):
            try:
                data = _load_json(path)
            except json.JSONDecodeError:
                continue
            vocabs[path.stem] = data
        return vocabs

    def _map_label(self, value, preferred_names):
        label = _clean_label(value)
        if label is None:
            return None
        if not label.isdigit():
            return label
        index = int(label)
        for name, vocab in self.vocabs.items():
            if not any(token in name.lower() for token in preferred_names):
                continue
            resolved = self._lookup_vocab(vocab, index)
            if resolved is not None:
                return resolved
        return label

    def _lookup_vocab(self, vocab, index):
        if isinstance(vocab, list) and 0 <= index < len(vocab):
            return _clean_label(vocab[index])
        if isinstance(vocab, dict):
            for key in (str(index), index):
                if key in vocab:
                    value = vocab[key]
                    if isinstance(value, dict):
                        value = _first(value, ("name", "label", "category", "attribute", "affordance"))
                    return _clean_label(value)
        return None

    def _extract_objects(self, image_record):
        objects = _first(image_record, self.object_keys)
        if objects is None:
            return [image_record] if _first(image_record, self.category_keys) is not None else []
        if isinstance(objects, dict):
            return list(objects.values())
        return _as_list(objects)

    def _extract_image_ref(self, image_record):
        value = _first(image_record, self.image_keys)
        if isinstance(value, dict):
            value = _first(value, self.image_keys)
        label = _clean_label(value)
        if label is not None:
            return label
        for value in image_record.values() if isinstance(image_record, dict) else []:
            label = _clean_label(value)
            if label and Path(label).suffix.lower() in IMAGE_SUFFIXES:
                return label
        return None

    def _extract_bbox(self, obj):
        box = _first(obj, self.bbox_keys)
        if isinstance(box, dict):
            keys = {key.lower(): value for key, value in box.items()}
            if {"xmin", "ymin", "xmax", "ymax"} <= set(keys):
                return [keys["xmin"], keys["ymin"], keys["xmax"], keys["ymax"]]
            if {"x", "y", "w", "h"} <= set(keys):
                return [keys["x"], keys["y"], keys["w"], keys["h"]]
        if box is None:
            return None
        box = list(box)
        if len(box) < 4:
            return None
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

    def _normalize_box(self, box, width, height):
        if box is None:
            return (0, 0, width, height)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(1.0, x2)
            y2 = y1 + max(1.0, y2)
        x1 = int(max(0, min(width - 1, round(x1))))
        y1 = int(max(0, min(height - 1, round(y1))))
        x2 = int(max(x1 + 1, min(width, round(x2))))
        y2 = int(max(y1 + 1, min(height, round(y2))))
        return (x1, y1, x2, y2)

    def _extract_labels(self, obj, keys, preferred_names):
        labels = []
        for value in _as_list(_first(obj, keys)):
            if isinstance(value, dict):
                value = _first(value, ("name", "label", "category", "attribute", "affordance", "value"))
            label = self._map_label(value, preferred_names)
            if label is not None and label not in labels:
                labels.append(label)
        return labels

    def _flatten_annotations(self):
        samples = []
        for image_index, image_record in enumerate(self.raw_annotations):
            if not isinstance(image_record, dict):
                continue
            image_ref = self._extract_image_ref(image_record)
            objects = self._extract_objects(image_record)
            for object_index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue
                category = self._map_label(_first(obj, self.category_keys), ("class", "category"))
                if category is None:
                    category = self._map_label(_first(image_record, self.category_keys), ("class", "category"))
                if category is None:
                    continue
                samples.append(
                    {
                        "image_ref": image_ref,
                        "image_index": image_index,
                        "object_index": object_index,
                        "category": category,
                        "bbox": self._extract_bbox(obj),
                        "attributes": self._extract_labels(obj, self.attribute_keys, ("attr", "attribute")),
                        "affordances": self._extract_labels(obj, self.affordance_keys, ("aff", "affordance")),
                    }
                )
        return samples

    def _build_image_index(self):
        search_roots = [self.data_dir, self.data_root]
        index = {}
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    index.setdefault(path.name, path)
                    index.setdefault(path.stem, path)
        self._image_index = index

    def resolve_image_path(self, image_ref):
        if image_ref is None:
            raise FileNotFoundError("Annotation has no image path/name field.")
        path = Path(image_ref)
        candidates = [
            path,
            self.data_root / path,
            self.data_dir / path,
            self.data_dir / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if self._image_index is None:
            self._build_image_index()
        for key in (path.name, path.stem, image_ref):
            if key in self._image_index:
                return self._image_index[key]
        raise FileNotFoundError(f"Could not resolve image path for annotation reference: {image_ref}")

    def _make_multihot(self, labels, mapping):
        target = torch.zeros(len(mapping), dtype=torch.float32)
        for label in labels:
            index = mapping.get(label)
            if index is not None:
                target[index] = 1.0
        return target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        item = dict(sample)
        item["attribute_target"] = self._make_multihot(sample["attributes"], self.attribute_to_index)
        item["affordance_target"] = self._make_multihot(sample["affordances"], self.affordance_to_index)

        if not self.load_images:
            return item

        image_path = self.resolve_image_path(sample["image_ref"])
        image = Image.open(image_path).convert("RGB")
        crop = image.crop(self._normalize_box(sample["bbox"], image.width, image.height))
        item["image_path"] = str(image_path)
        item["image"] = self.preprocess(crop) if self.preprocess is not None else crop
        if self.tokenizer is not None:
            item["category_tokens"] = self.tokenizer([sample["category"]])[0]
        return item


def collate_ocl_batch(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "categories": [item["category"] for item in batch],
        "attributes": [item["attributes"] for item in batch],
        "affordances": [item["affordances"] for item in batch],
        "attribute_targets": torch.stack([item["attribute_target"] for item in batch], dim=0),
        "affordance_targets": torch.stack([item["affordance_target"] for item in batch], dim=0),
        "image_paths": [item["image_path"] for item in batch],
        "object_indices": [item["object_index"] for item in batch],
    }
