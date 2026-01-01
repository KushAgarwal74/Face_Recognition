from pathlib import Path
from PIL import Image, ImageOps
import pillow_heif

# Enable HEIC support
pillow_heif.register_heif_opener()

RAW_ORIGINAL = Path("data/raw_original")
RAW = Path("data/raw")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".heic", ".png"}

def convert_to_png(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok= True)

    with Image.open(src_path) as img:
        print(img.getexif())
        img = ImageOps.exif_transpose(img)
        img.save(dst_path, format="PNG")

def prepare_dataset():
    for img_path in RAW_ORIGINAL.rglob("*"):
        if img_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        
        relative_path = img_path.relative_to(RAW_ORIGINAL)
        png_path = (RAW / relative_path).with_suffix(".png")

        if png_path.exists():
            continue

        print(f"Converting: {img_path} -> {png_path}")
        convert_to_png(img_path, png_path)

if __name__ == "__main__":
    prepare_dataset()

# src_path.suffix
# src_path.exists()
# src_path.parent
# src_path.name
# src_path.convert()
