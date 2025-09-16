import argparse
from pathlib import Path

try:
    from yolo11_tf.data import write_tfrecords_from_yaml
except ModuleNotFoundError:
    # Allow running this script directly: add repo root to sys.path
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from yolo11_tf.data import write_tfrecords_from_yaml


def parse_args():
    ap = argparse.ArgumentParser("Create TFRecords from YOLO-format dataset YAML")
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--out", type=str, required=True, help="Output directory for TFRecords")
    ap.add_argument("--shards", type=int, default=8, help="Number of shards per split")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Writing train TFRecords...")
    write_tfrecords_from_yaml(args.data, str(out), split='train', shards=args.shards)
    print("Writing val TFRecords...")
    write_tfrecords_from_yaml(args.data, str(out), split='val', shards=max(1, args.shards // 2))
    print(f"Done. TFRecords stored under {out}")


if __name__ == "__main__":
    main()
