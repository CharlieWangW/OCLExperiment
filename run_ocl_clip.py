import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from ocl_clip.data import OCLDataset, collate_ocl_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP inference on OCL.")
    parser.add_argument("--data-root", required=True, help="Path to OCL_data.")
    parser.add_argument("--split", default="valtest", help="OCL split name.")
    parser.add_argument("--clip-model", default="ViT-B/32", help="OpenAI CLIP model.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-categories", type=int, default=10)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="outputs/ocl_clip_metrics.json")
    return parser.parse_args()


def freeze_model(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def encode_texts(model, tokenizer, labels, device, prompt_template):
    if not labels:
        return torch.empty((0, model.text_projection.shape[1]), dtype=torch.float32, device=device)
    prompts = [prompt_template.format(label=label.replace("_", " ")) for label in labels]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def topk_correct(logits, target, topk):
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1))
    return {k: correct[: min(k, max_k)].any(dim=0).float().sum().item() for k in topk}


def multilabel_map(score_rows, target_rows):
    if not score_rows:
        return float("nan")
    scores = np.asarray(score_rows, dtype=np.float32)
    targets = np.asarray(target_rows, dtype=np.int32)
    ap_values = []
    for col in range(targets.shape[1]):
        positives = int(targets[:, col].sum())
        negatives = int((1 - targets[:, col]).sum())
        if positives == 0 or negatives == 0:
            continue
        ap_values.append(average_precision_score(targets[:, col], scores[:, col]))
    return float(np.mean(ap_values)) if ap_values else float("nan")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model, preprocess = clip.load(args.clip_model, device=device)
    freeze_model(model)

    dataset = OCLDataset(
        data_root=args.data_root,
        split=args.split,
        preprocess=preprocess,
        tokenizer=clip.tokenize,
        top_categories=args.top_categories,
        max_samples=args.max_samples,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_ocl_batch,
    )

    category_text = encode_texts(
        model,
        clip.tokenize,
        dataset.selected_categories,
        device,
        "a photo of a {label}.",
    )
    attribute_text = encode_texts(
        model,
        clip.tokenize,
        dataset.attribute_labels,
        device,
        "a photo of an object that is {label}.",
    )
    affordance_text = encode_texts(
        model,
        clip.tokenize,
        dataset.affordance_labels,
        device,
        "a photo of an object used for {label}.",
    )

    category_to_index = {name: idx for idx, name in enumerate(dataset.selected_categories)}
    topk_totals = {k: 0.0 for k in args.topk}
    total = 0
    attr_scores, attr_targets = [], []
    aff_scores, aff_targets = [], []

    for batch in tqdm(loader, desc="CLIP inference"):
        images = batch["images"].to(device, non_blocking=True)
        targets = torch.tensor(
            [category_to_index[label] for label in batch["categories"]],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            category_logits = image_features @ category_text.t()
            attribute_logits = image_features @ attribute_text.t()
            affordance_logits = image_features @ affordance_text.t()

        correct = topk_correct(category_logits, targets, args.topk)
        for k, value in correct.items():
            topk_totals[k] += value
        total += images.shape[0]

        attr_scores.extend(attribute_logits.detach().cpu().float().numpy().tolist())
        aff_scores.extend(affordance_logits.detach().cpu().float().numpy().tolist())
        attr_targets.extend(batch["attribute_targets"].cpu().int().numpy().tolist())
        aff_targets.extend(batch["affordance_targets"].cpu().int().numpy().tolist())

    metrics = {
        "data_root": str(Path(args.data_root).resolve()),
        "split": args.split,
        "clip_model": args.clip_model,
        "num_object_samples": total,
        "selected_categories": dataset.selected_categories,
        "category_topk_accuracy": {f"top{k}": topk_totals[k] / max(total, 1) for k in args.topk},
        "attribute_mAP": multilabel_map(attr_scores, attr_targets),
        "affordance_mAP": multilabel_map(aff_scores, aff_targets),
        "num_attributes": len(dataset.attribute_labels),
        "num_affordances": len(dataset.affordance_labels),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
