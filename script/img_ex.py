from pathlib import Path
from PIL import Image, ImageOps, ExifTags

src_path = Path("data/raw_original/subject_1/0.jpg")

with Image.open(src_path) as img:
    print("Before:", img.getexif())
    
    exif = img.getexif()
    for exif_tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(exif_tag_id, exif_tag_id)
        print(f"{tag_name}: {value}")

    img = ImageOps.exif_transpose(img)
    print("After:", img.getexif())

