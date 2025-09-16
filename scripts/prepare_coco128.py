import argparse
from pathlib import Path
import zipfile
from urllib.request import urlopen, Request

# Allow running directly by ensuring repo root is on sys.path for yolo11_tf imports if needed
try:
    import yolo11_tf  # noqa: F401
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"


def download(url: str, dest: Path, chunk: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
    tmp.replace(dest)


def safe_extract_zip(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            name = m.filename
            dest = (target_dir / name).resolve()
            if not str(dest).startswith(str(target_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {name}")
        z.extractall(target_dir)


def write_yaml(yaml_path: Path, root: Path, names):
    import yaml
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train2017",
        "val": "images/train2017",
        "names": names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)


def prepare(out_dir: Path):
    out_dir = out_dir.resolve()
    downloads = out_dir / "_downloads"
    z = downloads / "coco128.zip"
    if not z.exists():
        print("Downloading coco128...")
        download(URL, z)
    else:
        print("coco128.zip already exists")
    if not (out_dir / "coco128").exists():
        print("Extracting coco128...")
        safe_extract_zip(z, out_dir)
    else:
        print("coco128 already extracted")
    names = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant',
        'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
        'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
        'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
        'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]
    yaml_path = out_dir / "coco128.yaml"
    write_yaml(yaml_path, out_dir / "coco128", names)
    print(f"Wrote dataset YAML: {yaml_path}")
    return yaml_path


def parse_args():
    ap = argparse.ArgumentParser("Download and prepare coco128 test dataset")
    ap.add_argument("--out", type=str, default=str(Path("datasets")), help="Target base directory")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(Path(args.out))
