from __future__ import annotations

import argparse
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from PIL import Image


# detection-datasets/coco 的 objects.category 使用 0-79 的 COCO 80 类索引。
COCO_80_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Hugging Face 自动导入 COCO 小样本。")
    parser.add_argument("--dataset", default="detection-datasets/coco", help="Hugging Face 数据集名称。")
    parser.add_argument("--split", default="val", help="数据集 split，默认 val。")
    parser.add_argument("--max-images", type=int, default=100, help="最多导入多少张图片。")
    parser.add_argument("--balanced", action="store_true", help="按 COCO 80 类均衡采样。")
    parser.add_argument("--per-category", type=int, default=100, help="均衡模式下每个类别保存多少张。")
    parser.add_argument("--max-scan", type=int, default=0, help="均衡模式最多扫描多少条数据；0 表示不限制。")
    parser.add_argument("--output-dir", default="images", help="图片输出目录，相对 backend。")
    parser.add_argument("--labels-file", default="labels.txt", help="标签文件路径，相对 backend。")
    parser.add_argument("--prefix", default="coco", help="保存图片时使用的文件名前缀。")
    parser.add_argument("--streaming", action="store_true", default=True, help="使用 streaming 模式读取大数据集。")
    parser.add_argument("--overwrite-labels", action="store_true", help="覆盖 labels.txt，而不是追加去重。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = (base_dir / args.output_dir).resolve()
    labels_file = (base_dir / args.labels_file).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} split={args.split} from Hugging Face...")
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    if args.balanced:
        prepare_balanced_coco(dataset, args, output_dir, labels_file)
        return

    prepare_first_n_coco(dataset, args, output_dir, labels_file)


def prepare_first_n_coco(dataset, args: argparse.Namespace, output_dir: Path, labels_file: Path) -> None:
    saved_count = 0
    labels = [] if args.overwrite_labels else read_existing_labels(labels_file)
    seen_labels = set(labels)

    for index, row in enumerate(dataset):
        if saved_count >= args.max_images:
            break

        image = extract_image(row)
        if image is None:
            continue

        image_id = row.get("image_id") or row.get("id") or index
        filename = f"{safe_filename(args.prefix)}_{args.split}_{safe_filename(str(image_id))}.jpg"
        image_path = output_dir / filename
        image.convert("RGB").save(image_path, format="JPEG", quality=92)
        saved_count += 1

        for label in extract_labels(row):
            if label not in seen_labels:
                labels.append(label)
                seen_labels.add(label)

        print(f"[{saved_count}/{args.max_images}] saved {image_path.name}")

    write_labels(labels_file, labels)
    print("")
    print(f"Done. Images saved: {saved_count}")
    print(f"Labels written: {len(labels)}")
    print("Next: start backend or call POST /rebuild_index to rebuild FAISS indexes.")


def prepare_balanced_coco(dataset, args: argparse.Namespace, output_dir: Path, labels_file: Path) -> None:
    if args.per_category <= 0:
        raise ValueError("--per-category 必须大于 0。")

    labels = [] if args.overwrite_labels else read_existing_labels(labels_file)
    seen_labels = set(labels)
    for label in COCO_80_LABELS:
        if label not in seen_labels:
            labels.append(label)
            seen_labels.add(label)

    category_counts = count_existing_balanced_images(output_dir, args.prefix, args.split)
    saved_image_ids = set()
    scanned_count = 0
    saved_count = 0
    target_total = len(COCO_80_LABELS) * args.per_category

    for index, row in enumerate(dataset):
        scanned_count += 1
        if args.max_scan and scanned_count > args.max_scan:
            break

        if all(category_counts[label] >= args.per_category for label in COCO_80_LABELS):
            break

        image_id = str(row.get("image_id") or row.get("id") or index)
        if image_id in saved_image_ids:
            continue

        categories = extract_category_names(row)
        candidates = [label for label in categories if category_counts[label] < args.per_category]
        if not candidates:
            continue

        target_label = min(candidates, key=lambda label: category_counts[label])
        image = extract_image(row)
        if image is None:
            continue

        target_dir = output_dir / safe_filename(target_label)
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_filename(args.prefix)}_{args.split}_{safe_filename(target_label)}_{safe_filename(image_id)}.jpg"
        image_path = target_dir / filename
        if not image_path.exists():
            image.convert("RGB").save(image_path, format="JPEG", quality=92)
            saved_count += 1

        category_counts[target_label] += 1
        saved_image_ids.add(image_id)

        if saved_count == 1 or saved_count % 50 == 0:
            filled = sum(min(count, args.per_category) for count in category_counts.values())
            print(f"saved={saved_count}, filled={filled}/{target_total}, scanned={scanned_count}")

    write_labels(labels_file, labels)
    print("")
    print(f"Done. Newly saved images: {saved_count}")
    print(f"Scanned rows: {scanned_count}")
    print(f"Labels written: {len(labels)}")
    print_balanced_summary(category_counts, args.per_category)
    print("Next: start backend or call POST /rebuild_index to rebuild FAISS indexes.")


def extract_image(row: dict) -> Image.Image | None:
    image = row.get("image")
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, dict):
        if image.get("bytes"):
            return Image.open(BytesIO(image["bytes"]))
        if image.get("path"):
            return Image.open(image["path"])

    if isinstance(image, (str, Path)):
        return Image.open(image)

    return None


def extract_labels(row: dict) -> list[str]:
    labels: list[str] = []

    for key in ("caption", "caption_en", "text", "sentence"):
        value = row.get(key)
        labels.extend(normalize_text_values(value))

    objects = row.get("objects") or row.get("annotations")
    if isinstance(objects, dict):
        for key in ("category_name", "category_names", "label", "labels"):
            labels.extend(normalize_text_values(objects.get(key)))

        for key in ("category", "category_id"):
            labels.extend(category_ids_to_names(objects.get(key)))

    return deduplicate(labels)


def extract_category_names(row: dict) -> list[str]:
    objects = row.get("objects") or row.get("annotations")
    if not isinstance(objects, dict):
        return []

    labels = []
    labels.extend(normalize_text_values(objects.get("category_name")))
    labels.extend(normalize_text_values(objects.get("category_names")))
    labels.extend(normalize_text_values(objects.get("label")))
    labels.extend(normalize_text_values(objects.get("labels")))
    labels.extend(category_ids_to_names(objects.get("category")))
    labels.extend(category_ids_to_names(objects.get("category_id")))
    return deduplicate(label for label in labels if label in COCO_80_LABELS)


def normalize_text_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        labels = []
        for item in value:
            if isinstance(item, str) and item.strip():
                labels.append(item.strip())
        return labels
    return []


def category_ids_to_names(value) -> list[str]:
    if value is None:
        return []

    values = value if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) else [value]
    labels = []
    for item in values:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue

        if 0 <= index < len(COCO_80_LABELS):
            labels.append(COCO_80_LABELS[index])
        else:
            labels.append(str(index))
    return labels


def count_existing_balanced_images(output_dir: Path, prefix: str, split: str) -> dict[str, int]:
    counts = {label: 0 for label in COCO_80_LABELS}
    for label in COCO_80_LABELS:
        category_dir = output_dir / safe_filename(label)
        if not category_dir.exists():
            continue
        pattern = f"{safe_filename(prefix)}_{split}_{safe_filename(label)}_*.jpg"
        counts[label] = len(list(category_dir.glob(pattern)))
    return counts


def print_balanced_summary(category_counts: dict[str, int], per_category: int) -> None:
    missing = {
        label: per_category - count
        for label, count in category_counts.items()
        if count < per_category
    }
    completed = len(COCO_80_LABELS) - len(missing)
    print(f"Completed categories: {completed}/{len(COCO_80_LABELS)}")
    if missing:
        print("Categories still below target:")
        for label, count in sorted(missing.items(), key=lambda item: item[1], reverse=True):
            print(f"  {label}: missing {count}")


def read_existing_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    return deduplicate(line.strip() for line in path.read_text(encoding="utf-8").splitlines())


def write_labels(path: Path, labels: list[str]) -> None:
    path.write_text("\n".join(deduplicate(labels)) + "\n", encoding="utf-8")


def deduplicate(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        item = value.strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "item"


if __name__ == "__main__":
    main()
