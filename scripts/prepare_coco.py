import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import zipfile


COCO_TRAIN_ZIP = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.1f} {units[i]}"


def download(url: str, dest: Path, chunk: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as r, open(tmp, "wb") as f:
            total = r.length or 0
            t0 = time.time()
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100.0 * done / total
                    rate = done / max(1e-6, (time.time() - t0))
                    print(f"  {pct:6.2f}%  {human_bytes(done):>10} / {human_bytes(total):<10}  {human_bytes(int(rate))}/s",
                          end="\r", flush=True)
        print()
        tmp.replace(dest)
    except (URLError, HTTPError) as e:
        print(f"Failed to download {url}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def safe_extract_zip(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            name = m.filename
            # Prevent path traversal
            dest = (target_dir / name).resolve()
            if not str(dest).startswith(str(target_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {name}")
        z.extractall(target_dir)


def coco_to_yolo_labels(instances_json: Path, labels_out: Path):
    labels_out.mkdir(parents=True, exist_ok=True)
    with open(instances_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build image id -> (file_name, width, height)
    img_map = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in data["images"]}
    # Build contiguous category id mapping and names
    cats = sorted(data["categories"], key=lambda x: x["id"])  # order by COCO id
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    # Group annotations per image
    anns_per_image = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        if img_id not in img_map:
            continue
        anns_per_image.setdefault(img_id, []).append(ann)

    # Write labels
    num_boxes = 0
    for img_id, (file_name, w, h) in img_map.items():
        anns = anns_per_image.get(img_id, [])
        if not anns:
            # no file is fine; training code handles missing label files
            continue
        stem = Path(file_name).with_suffix("").name
        out_path = labels_out / f"{stem}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for a in anns:
                x, y, bw, bh = a["bbox"]
                # clip bbox to image
                x1 = max(0.0, min(x, w - 1.0))
                y1 = max(0.0, min(y, h - 1.0))
                x2 = max(0.0, min(x + bw, w - 1.0))
                y2 = max(0.0, min(y + bh, h - 1.0))
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if bw <= 1e-6 or bh <= 1e-6:
                    continue
                cx = x1 + bw / 2.0
                cy = y1 + bh / 2.0
                cxn = cx / float(w)
                cyn = cy / float(h)
                bwn = bw / float(w)
                bhn = bh / float(h)
                cls = cat_id_to_idx.get(a["category_id"])  # 0..79
                if cls is None:
                    continue
                f.write(f"{cls} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f}\n")
                num_boxes += 1

    return names, num_boxes


def write_yaml(yaml_path: Path, root: Path, names):
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train2017",
        "val": "images/val2017",
        "names": names,
    }
    import yaml
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)


def prepare_coco(out_dir: Path):
    out_dir = out_dir.resolve()
    downloads = out_dir / "_downloads"
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    ann_dir = out_dir  # annotations zip extracts to out_dir/annotations

    print(f"Preparing COCO2017 at: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download zips
    train_zip = downloads / "train2017.zip"
    val_zip = downloads / "val2017.zip"
    ann_zip = downloads / "annotations_trainval2017.zip"
    if not train_zip.exists():
        print("Downloading train2017 images (~18GB)...")
        download(COCO_TRAIN_ZIP, train_zip)
    else:
        print("train2017.zip already exists, skipping download")
    if not val_zip.exists():
        print("Downloading val2017 images (~1GB)...")
        download(COCO_VAL_ZIP, val_zip)
    else:
        print("val2017.zip already exists, skipping download")
    if not ann_zip.exists():
        print("Downloading annotations (~250MB)...")
        download(COCO_ANN_ZIP, ann_zip)
    else:
        print("annotations_trainval2017.zip already exists, skipping download")

    # 2) Extract
    images_dir.mkdir(parents=True, exist_ok=True)
    if not (images_dir / "train2017").exists():
        print("Extracting train2017 images...")
        # The zip contains a top-level 'train2017/' folder with images
        # Extract directly under images_dir so we end up with images/train2017/
        safe_extract_zip(train_zip, images_dir)
    else:
        print("images/train2017 already exists, skipping extract")

    if not (images_dir / "val2017").exists():
        print("Extracting val2017 images...")
        # The zip contains a top-level 'val2017/' folder with images
        safe_extract_zip(val_zip, images_dir)
    else:
        print("images/val2017 already exists, skipping extract")

    if not (ann_dir / "annotations" / "instances_train2017.json").exists():
        print("Extracting annotations...")
        safe_extract_zip(ann_zip, out_dir)
        # annotations/ is created at out_dir by the zip; no rename needed
    else:
        print("annotations already exists, skipping extract")

    # 3) Convert to YOLO labels
    labels_train = labels_dir / "train2017"
    labels_val = labels_dir / "val2017"
    train_json = ann_dir / "annotations" / "instances_train2017.json"
    val_json = ann_dir / "annotations" / "instances_val2017.json"

    print("Converting COCO train annotations to YOLO...")
    names, n_train = coco_to_yolo_labels(train_json, labels_train)
    print(f"  Wrote {n_train} boxes for train2017")
    print("Converting COCO val annotations to YOLO...")
    names_val, n_val = coco_to_yolo_labels(val_json, labels_val)
    print(f"  Wrote {n_val} boxes for val2017")

    # 4) Write dataset YAML
    yaml_path = out_dir.parent / "coco2017.yaml"
    write_yaml(yaml_path, out_dir, names)
    print(f"Wrote dataset YAML: {yaml_path}")

    # 5) Cleanup downloads (optional)
    print("Done. You may remove downloads to free space:", downloads)


def parse_args():
    ap = argparse.ArgumentParser("Download and prepare COCO2017 in YOLO format")
    ap.add_argument("--out", type=str, default=str(Path("datasets") / "coco2017"),
                    help="Target dataset root directory")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_coco(Path(args.out))
