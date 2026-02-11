"""
Data loading and parsing utilities for AITEX Fabric Dataset.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import cv2


# Defect code to name mapping
DEFECT_CODES = {
    0: "No Defect",
    2: "Broken end",
    6: "Broken yarn",
    10: "Broken pick",
    16: "Weft curling",
    19: "Fuzzyball",
    22: "Cut selvage",
    23: "Crease",
    25: "Warp ball",
    27: "Knots",
    29: "Contamination",
    30: "Nep",
    36: "Weft crack"
}

# Base data directory
BASE_DIR = Path(__file__).parent.parent / "data" / "AITEX_Fabric_Image_Database"


def parse_filename(filename: str) -> dict[str, int]:
    """
    Parse image filename to extract metadata.

    Format: nnnn_ddd_ff.png
    - nnnn: image number
    - ddd: defect code (000 for no defect)
    - ff: fabric code

    Args:
        filename: Image filename (e.g., '0001_006_02.png')

    Returns:
        Dictionary with keys: 'image_id', 'defect_code', 'fabric_code'
    """
    name = Path(filename).stem.replace('_mask', '')
    parts = name.split('_')

    return {
        'image_id': int(parts[0]),
        'defect_code': int(parts[1]),
        'fabric_code': int(parts[2])
    }


def get_defect_name(defect_code: int) -> str:
    """Get defect name from code."""
    return DEFECT_CODES.get(defect_code, f"Unknown ({defect_code})")


def load_dataset_info() -> pd.DataFrame:
    """
    Scan dataset directories and create a DataFrame with all image information.

    Returns:
        DataFrame with columns: image_id, defect_code, fabric_code, defect_name,
                               has_defect, image_path, mask_path
    """
    data = []

    # Load defect images
    defect_dir = BASE_DIR / "Defect_images"
    mask_dir = BASE_DIR / "Mask_images"

    if defect_dir.exists():
        for img_file in defect_dir.glob("*.png"):
            info = parse_filename(img_file.name)
            mask_file = mask_dir / f"{img_file.stem}_mask.png"

            data.append({
                **info,
                'defect_name': get_defect_name(info['defect_code']),
                'has_defect': True,
                'image_path': str(img_file),
                'mask_path': str(mask_file) if mask_file.exists() else None
            })

    # Load non-defect images
    nodefect_dir = BASE_DIR / "NODefect_images"
    if nodefect_dir.exists():
        for img_file in nodefect_dir.rglob("*.png"):
            info = parse_filename(img_file.name)

            data.append({
                **info,
                'defect_name': get_defect_name(0),
                'has_defect': False,
                'image_path': str(img_file),
                'mask_path': None
            })

    df = pd.DataFrame(data)
    return df.sort_values('image_id').reset_index(drop=True)


def load_image(image_path: str, as_rgb: bool = True) -> np.ndarray:
    """
    Load image from path.

    Args:
        image_path: Path to image file
        as_rgb: If True, return RGB; if False, return grayscale

    Returns:
        Image as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def load_mask(mask_path: str) -> np.ndarray:
    """
    Load binary mask image.

    Args:
        mask_path: Path to mask file

    Returns:
        Binary mask as numpy array (0 or 255)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return mask


def crop_defect_from_mask(image: np.ndarray, mask: np.ndarray,
                          padding: int = 10) -> Optional[np.ndarray]:
    """
    Extract defect region from image using mask bounding box.

    Args:
        image: Original image
        mask: Binary mask
        padding: Pixels to add around defect bbox

    Returns:
        Cropped defect region or None if mask is empty
    """
    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get bounding box of all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Add padding
    h, w = image.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Crop
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped


def load_image_triplet(image_path: str, mask_path: str,
                       padding: int = 10) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load original image, mask, and cropped defect region.

    Args:
        image_path: Path to defect image
        mask_path: Path to mask image
        padding: Pixels to add around defect bbox

    Returns:
        Tuple of (original_image, mask, cropped_defect)
    """
    image = load_image(image_path, as_rgb=True)
    mask = load_mask(mask_path)
    cropped = crop_defect_from_mask(image, mask, padding=padding)

    return image, mask, cropped


def get_mask_bbox(mask: np.ndarray) -> Optional[dict[str, int]]:
    """
    Get bounding box coordinates from mask.

    Args:
        mask: Binary mask

    Returns:
        Dictionary with keys: x, y, width, height, or None if mask is empty
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get bounding box of all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    return {
        'x': int(x_min),
        'y': int(y_min),
        'width': int(x_max - x_min),
        'height': int(y_max - y_min)
    }


def get_sample_images(df: pd.DataFrame, n_samples: int = 12,
                     defects_only: bool = True) -> pd.DataFrame:
    """
    Get sample images from dataset.

    Args:
        df: Dataset DataFrame
        n_samples: Number of samples to return
        defects_only: If True, return only defect images

    Returns:
        DataFrame with sampled images
    """
    if defects_only:
        df = df[df['has_defect']]

    # Try to get diverse samples across defect types
    if 'defect_code' in df.columns and len(df['defect_code'].unique()) > 1:
        samples = df.groupby('defect_code').apply(
            lambda x: x.sample(min(2, len(x)), random_state=42)
        ).reset_index(drop=True)

        # If we need more samples, add random ones
        if len(samples) < n_samples:
            remaining = df[~df.index.isin(samples.index)]
            additional = remaining.sample(
                min(n_samples - len(samples), len(remaining)),
                random_state=42
            )
            samples = pd.concat([samples, additional]).reset_index(drop=True)
    else:
        samples = df.sample(min(n_samples, len(df)), random_state=42)

    return samples.head(n_samples)
