import os
import torch
import numpy as np
import cv2
import argparse
import jpegio as jio
import pickle
from PIL import Image
import torchvision.transforms as transforms
from dtd import seg_dtd, recursive_replace_gelu

from swins import (
    BasicLayer,
    SwinTransformerBlock,
    PatchMerging,
    PatchEmbed,
    WindowAttention,
    Mlp
)

parser = argparse.ArgumentParser(description="Run tampering inference and save an overlay.")
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input JPEG images')
parser.add_argument('--pth', type=str, default='models/dtd_doctamper.pth', help='Path to model checkpoint')
parser.add_argument('--mask_output_dir', type=str, default='output_masks', help='Directory to save binary tamper masks')
parser.add_argument('--overlay_output_dir', type=str, default='output_overlays', help='Directory to save tamper overlays on the original images')
parser.add_argument('--qt_path', type=str, default='qt_table.pk', help='Path to quantization table pickle')
parser.add_argument('--overlay_alpha', type=float, default=0.45, help='Opacity of the tamper overlay (0-1)')
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS) acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Using CPU.")

def load_qt_table(path):
    print(f"Loading quantization table from {path}...")
    with open(path, 'rb') as f:
        qt_data = pickle.load(f)
    return qt_data

def get_closest_quality_id(image_qt, qt_dict):
    target_qt = image_qt.flatten()
    best_id = 90
    min_diff = float('inf')

    for q_id, table in qt_dict.items():
        ref_qt = np.array(table).flatten()
        if len(ref_qt) != len(target_qt):
            continue
        diff = np.sum(np.abs(target_qt - ref_qt))
        if diff < min_diff:
            min_diff = diff
            best_id = q_id
        if diff == 0:
            break

    print(f"Matched JPEG Quality ID: {best_id} (Diff: {min_diff})")
    return best_id

def load_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    model = seg_dtd('', 2)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    recursive_replace_gelu(model)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, qt_dict):
    im = Image.open(image_path).convert('RGB')
    orig_w, orig_h = im.size

    stride = 64
    new_w = (orig_w // stride) * stride
    new_h = (orig_h // stride) * stride
    if new_w != orig_w or new_h != orig_h:
        print(f"Resizing image from {orig_w}x{orig_h} to {new_w}x{new_h} (must be divisible by {stride})")
        im = im.resize((new_w, new_h), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img_tensor = transform(im).unsqueeze(0).to(device)

    jpeg = jio.read(image_path)

    dct_coef = jpeg.coef_arrays[0].copy()

    target_dct_h = new_h
    target_dct_w = new_w

    curr_dct_h, curr_dct_w = dct_coef.shape

    new_dct = np.zeros((target_dct_h, target_dct_w))

    h_copy = min(curr_dct_h, target_dct_h)
    w_copy = min(curr_dct_w, target_dct_w)

    new_dct[:h_copy, :w_copy] = dct_coef[:h_copy, :w_copy]

    dct_tensor = torch.tensor(np.clip(np.abs(new_dct), 0, 20)).long().unsqueeze(0).to(device)

    if jpeg.quant_tables:
        raw_qt = jpeg.quant_tables[0].copy()
        q_id = get_closest_quality_id(raw_qt, qt_dict)
        best_qt_values = qt_dict[q_id]
    else:
        print("default to 90")
        best_qt_values = qt_dict[90]

    qt_arr = np.array(best_qt_values).reshape(8, 8)
    qs_tensor = torch.LongTensor(qt_arr).unsqueeze(0).unsqueeze(0).to(device)

    return img_tensor, dct_tensor, qs_tensor, (orig_w, orig_h)

def build_overlay(original_bgr, mask, alpha=0.45, color=(0, 0, 255)):
    alpha = max(0.0, min(alpha, 1.0))
    if original_bgr.shape[:2] != mask.shape:
        raise ValueError("Mask and image spatial dimensions must match for overlay.")

    mask_bool = mask.astype(bool)
    color_layer = np.zeros_like(original_bgr)
    color_layer[:] = color

    tinted = cv2.addWeighted(original_bgr, 1 - alpha, color_layer, alpha, 0)
    overlay = original_bgr.copy()
    overlay[mask_bool] = tinted[mask_bool]
    return overlay

def list_images(input_dir):
    allowed_exts = {'.jpg', '.jpeg'}
    images = []
    for name in sorted(os.listdir(input_dir)):
        if os.path.splitext(name)[1].lower() in allowed_exts:
            images.append(os.path.join(input_dir, name))
    return images

def run_inference():
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        return
    if not os.path.exists(args.qt_path):
        print(f"Error: Quantization table file not found at {args.qt_path}")
        return

    image_paths = list_images(args.input_dir)
    if not image_paths:
        print(f"No JPEG images found in {args.input_dir}")
        return

    os.makedirs(args.mask_output_dir, exist_ok=True)
    os.makedirs(args.overlay_output_dir, exist_ok=True)

    qt_dict = load_qt_table(args.qt_path)
    model = load_model(args.pth)

    for image_path in image_paths:
        original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_bgr is None:
            print(f"Warning: Failed to read image at {image_path}; skipping.")
            continue
        orig_h, orig_w = original_bgr.shape[:2]

        result = preprocess_image(image_path, qt_dict)
        if result is None:
            print(f"Warning: Preprocessing failed for {image_path}; skipping.")
            continue
        img, dct, qs, (orig_w_pre, orig_h_pre) = result
        if (orig_w_pre, orig_h_pre) != (orig_w, orig_h):
            print("Warning: PIL and OpenCV disagree on image size; using PIL dimensions for resizing.")
            orig_w, orig_h = orig_w_pre, orig_h_pre
            original_bgr = cv2.resize(original_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        print(f"Running inference on {os.path.basename(image_path)}...")
        with torch.no_grad():
            output = model(img, dct, qs)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

            if pred_mask.shape != (orig_h, orig_w):
                pred_mask = cv2.resize(pred_mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            binary_mask = (pred_mask > 0.5).astype(np.uint8)
            mask_img = (binary_mask * 255).astype(np.uint8)

            stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(args.mask_output_dir, f"{stem}_mask.png")
            overlay_path = os.path.join(args.overlay_output_dir, f"{stem}_overlay.png")

            cv2.imwrite(mask_path, mask_img)
            overlay = build_overlay(original_bgr, binary_mask, alpha=args.overlay_alpha)
            cv2.imwrite(overlay_path, overlay)

            print(f"Saved mask -> {mask_path}")
            print(f"Saved overlay -> {overlay_path}")

if __name__ == '__main__':
    run_inference()
