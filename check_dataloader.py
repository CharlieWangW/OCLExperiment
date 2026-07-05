import argparse

import torch
from torchvision import transforms

from ocl_clip.data import OCLDataset, collate_ocl_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Check OCL Dataset and DataLoader without loading CLIP.")
    parser.add_argument("--data-root", default="toy_ocl_data")
    parser.add_argument("--split", default="valtest")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-categories", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = OCLDataset(
        data_root=args.data_root,
        split=args.split,
        preprocess=preprocess,
        top_categories=args.top_categories,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ocl_batch,
    )
    batch = next(iter(loader))
    print(f"samples: {len(dataset)}")
    print(f"selected categories: {dataset.selected_categories}")
    print(f"attribute labels: {dataset.attribute_labels}")
    print(f"affordance labels: {dataset.affordance_labels}")
    print(f"batch image tensor: {tuple(batch['images'].shape)}")
    print(f"batch categories: {batch['categories']}")
    print(f"batch image paths: {batch['image_paths']}")


if __name__ == "__main__":
    main()
