from pathlib import Path
import json
import shutil
from PIL import Image

CATEGORY_ID_TO_NAME = {
    1: "short_sleeved_shirt",
    2: "long_sleeved_shirt",
    3: "short_sleeved_outwear",
    4: "long_sleeved_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeved_dress",
    11: "long_sleeved_dress",
    12: "vest_dress",
    13: "sling_dress",
}

CATEGORY_ID_TO_YOLO = {k: k - 1 for k in CATEGORY_ID_TO_NAME.keys()}


def xyxy_to_yolo(box, img_w, img_h):
    """
    box: [x1, y1, x2, y2]
    returns normalized YOLO bbox: xc, yc, w, h
    """
    x1, y1, x2, y2 = box

    # clip to image
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return (
        xc / img_w,
        yc / img_h,
        bw / img_w,
        bh / img_h,
    )


def extract_items(annotation_dict):
    """
    DeepFashion2 per-image annotations often store items as:
      item1, item2, item3, ...
    and each item may contain category_id and bounding_box.
    """
    items = []
    for key, value in annotation_dict.items():
        if key.startswith("item") and isinstance(value, dict):
            items.append(value)
    return items


def convert_split(
    image_dir: Path,
    ann_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    copy_images: bool = True,
):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    converted = 0
    skipped = 0

    for img_path in image_paths:
        ann_path = ann_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            skipped += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        items = extract_items(ann)
        yolo_lines = []

        for item in items:
            category_id = item.get("category_id")
            bbox = item.get("bounding_box")

            if category_id not in CATEGORY_ID_TO_YOLO:
                continue
            if not bbox or len(bbox) != 4:
                continue

            yolo_box = xyxy_to_yolo(bbox, img_w, img_h)
            if yolo_box is None:
                continue

            cls = CATEGORY_ID_TO_YOLO[category_id]
            xc, yc, bw, bh = yolo_box
            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_labels_dir / f"{img_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        if copy_images:
            shutil.copy2(img_path, out_images_dir / img_path.name)

        converted += 1

    print(f"Converted: {converted}")
    print(f"Skipped (missing annotation): {skipped}")


def write_dataset_yaml(output_root: Path):
    yaml_text = f"""path: {output_root.resolve()}
train: train/images
val: validation/images

names:
"""
    for idx in range(len(CATEGORY_ID_TO_NAME)):
        name = CATEGORY_ID_TO_NAME[idx + 1]
        yaml_text += f"  {idx}: {name}\n"

    with open(output_root / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)


if __name__ == "__main__":
    root = Path("deepfashion2")

    train_images = root / "train" / "image"
    train_anns = root / "train" / "annos"

    val_images = root / "validation" / "image"
    val_anns = root / "validation" / "annos"

    test_images = root / "test" / "image"
    test_anns = root / "test" / "annos"

    out_root = Path("deepfashion2_yolo")

    if not Path(out_root / "train").exists():
        convert_split(
            train_images,
            train_anns,
            out_root / "train" / "images",
            out_root / "train" / "labels",
        )
    else:
        print("Train exists. Skipped.")

    if not Path(out_root / "validation").exists():
        convert_split(
            val_images,
            val_anns,
            out_root / "validation" / "images",
            out_root / "validation" / "labels",
        )
    else:
        print("Validation exists. Skipped.")
    
    write_dataset_yaml(out_root)
    print("Done.")