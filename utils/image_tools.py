import cv2
import fitz
import numpy as np
from typing import Optional
import os
import hashlib


def cv_to_md5(image):
    image_bytes = image.tobytes()

    # Compute MD5 hash value
    md5_hash = hashlib.md5(image_bytes).hexdigest()
    return f'{md5_hash}.png'

def paper_image_to_local(paper_iamge, cache_dir):
    save_name = cv_to_md5(paper_iamge)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    fname = f"{cache_dir}/{save_name}"

    cv2.imwrite(fname, paper_iamge)
    return fname

def pdf_page_to_local(doc, page_num, dpi , cache_dir):
    image = pdf_page_to_image(doc, page_num, dpi)
    return paper_image_to_local(image, cache_dir)

def convert_bbox_to_high_dpi(bbox_72dpi, source_dpi=72, target_dpi=300):
    """Convert bbox coordinates from 72 DPI to target DPI."""
    scale_factor = target_dpi / source_dpi
    return [
        bbox_72dpi[0] * scale_factor,
        bbox_72dpi[1] * scale_factor,
        bbox_72dpi[2] * scale_factor,
        bbox_72dpi[3] * scale_factor
    ]
    
def pdf_page_to_image(doc, page_num: int, dpi: int = 300) -> np.ndarray:

    def convert_pixmap_to_cvimage(pix: fitz.Pixmap) -> Optional[np.ndarray]:
        """Convert a Fitz Pixmap to an OpenCV image format."""
        COLOR_CONVERSIONS = {
            4: (cv2.COLOR_RGBA2BGR, "RGBA"),
            3: (cv2.COLOR_RGB2BGR, "RGB")
        }

        if pix.n not in COLOR_CONVERSIONS:
            raise ValueError(f"Unsupported number of image channels: {pix.n}")

        conversion_code, color_mode = COLOR_CONVERSIONS[pix.n]
        img_buffer = np.frombuffer(pix.samples, dtype=np.uint8)
        img_buffer = img_buffer.reshape(pix.h, pix.w, pix.n)
        return cv2.cvtColor(img_buffer, conversion_code)

    page = doc.load_page(page_num)
    mat = fitz.Matrix(dpi/72, dpi/72)
    
    # Get the high-resolution pixmap for the specified area
    pix = page.get_pixmap(
        matrix=mat,
        colorspace="rgb",
        alpha=False
    )
    
    try:
        return convert_pixmap_to_cvimage(pix)
    except Exception as e:
        return None

def crop_pdf_image(doc, page_num, dpi=72):  # Default is 72, consistent with pdf_page_to_image
    img = pdf_page_to_image(doc, page_num, dpi=dpi)
    return img

def crop_cv_img(img, bbox):
    h ,w = img.shape[0:2]
    bbox = [int(coord) for coord in bbox]
    bbox = [bbox[0], min(bbox[1], h), min(bbox[2],w), min(bbox[3], h)]
    table_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return table_img

def draw_lines(image, lines, color):
    for line in lines:
        rx0, ry0, rx1, ry1 = [int(x) for x in line['bbox']]
        cv2.rectangle(image, (rx0, ry0), (rx1, ry1), color, 1)
        cv2.putText(image, line['type'] , (rx0, ry0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def crop_pdf_bbox_image(doc, page_num, bbox, dpi):
    
    img = crop_pdf_image(doc, page_num, dpi)
    img = crop_cv_img(img, bbox)
    return img