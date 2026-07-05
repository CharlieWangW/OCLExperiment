import argparse
import pprint

from ocl_clip.data import OCLDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect OCL annotation shape.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="valtest")
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = OCLDataset(
        data_root=args.data_root,
        split=args.split,
        preprocess=None,
        tokenizer=None,
        top_categories=None,
        load_images=False,
    )
    print(f"annotation file: {dataset.annotation_file}")
    print(f"raw images: {len(dataset.raw_annotations)}")
    print(f"object samples: {len(dataset.samples)}")
    print(f"top categories: {dataset.selected_categories}")
    print()
    for sample in dataset.samples[: args.limit]:
        pprint.pp(sample)


if __name__ == "__main__":
    main()
