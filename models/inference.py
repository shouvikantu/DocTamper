import os
import torch
import numpy as np
import cv2
import argparse
import jpegio as jio
import pickle
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from dtd import seg_dtd, recursive_replace_gelu

# --- FIX: Import ALL Swin classes to global namespace for unpickling ---
from swins import (
    BasicLayer, 
    SwinTransformerBlock, 
    PatchMerging, 
    PatchEmbed, 
    WindowAttention, 
    Mlp
)

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image (must be JPEG)')
parser.add_argument('--pth', type=str, default='models/dtd_doctamper.pth', help='Path to model checkpoint')
parser.add_argument('--output_path', type=str, default='output_mask.png', help='Path to save the result mask')
parser.add_argument('--qt_path', type=str, default='qt_table.pk', help='Path to quantization table pickle')
args = parser.parse_args()

# --- Device Setup (Mac M4 Support) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS) acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# --- Helper: Find Closest Quality Factor ---
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

# --- Model Loading ---
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

# --- Image Preprocessing ---
def preprocess_image(image_path, qt_dict):
    try:
        im = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Could not open image. {e}")
        return None
        
    orig_w, orig_h = im.size
    
    # Resize to be divisible by 64 (Model Constraint)
    stride = 64 
    new_w = (orig_w // stride) * stride
    new_h = (orig_h // stride) * stride
    if new_w != orig_w or new_h != orig_h:
        print(f"Resizing image from {orig_w}x{orig_h} to {new_w}x{new_h} (must be divisible by {stride})")
        im = im.resize((new_w, new_h), Image.BILINEAR)

    # RGB Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img_tensor = transform(im).unsqueeze(0).to(device)

    # Extract JPEG Artifacts
    try:
        jpeg = jio.read(image_path)
    except:
        print("Error: Failed to read JPEG structure. Ensure input is a valid .jpg file.")
        return None
        
    dct_coef = jpeg.coef_arrays[0].copy()
    
    # Handle DCT Resizing
    # --- FIX: DCT tensor must have SAME spatial dimensions as RGB input ---
    # (H, W), NOT (H/8, W/8)
    target_dct_h = new_h 
    target_dct_w = new_w 
    
    curr_dct_h, curr_dct_w = dct_coef.shape
    
    # Create canvas
    new_dct = np.zeros((target_dct_h, target_dct_w))
    
    # Crop or Pad
    h_copy = min(curr_dct_h, target_dct_h)
    w_copy = min(curr_dct_w, target_dct_w)
    
    new_dct[:h_copy, :w_copy] = dct_coef[:h_copy, :w_copy]
    
    # Correct DCT Shape: (1, H, W)
    # The FPH module will apply stride=8 internally to match the VPH backbone.
    dct_tensor = torch.tensor(np.clip(np.abs(new_dct), 0, 20)).long().unsqueeze(0).to(device)

    # Quantization Table Handling
    best_qt_values = None
    if jpeg.quant_tables:
        raw_qt = jpeg.quant_tables[0].copy()
        q_id = get_closest_quality_id(raw_qt, qt_dict)
        best_qt_values = qt_dict[q_id] 
    else:
        print("Warning: No quantization table found. Defaulting to ID 90.")
        best_qt_values = qt_dict[90]
        
    qt_arr = np.array(best_qt_values).reshape(8, 8)
    qs_tensor = torch.LongTensor(qt_arr).unsqueeze(0).unsqueeze(0).to(device)

    return img_tensor, dct_tensor, qs_tensor, (orig_w, orig_h)

# --- Main Inference ---
def run_inference():
    if not os.path.exists(args.qt_path):
        print(f"Error: Quantization table file not found at {args.qt_path}")
        return

    qt_dict = load_qt_table(args.qt_path)
    model = load_model(args.pth)
    
    result = preprocess_image(args.image_path, qt_dict)
    if result is None:
        return
    img, dct, qs, (orig_w, orig_h) = result

    print("Running inference...")
    with torch.no_grad():
        output = model(img, dct, qs)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
        
        if pred_mask.shape != (orig_h, orig_w):
            pred_mask = cv2.resize(pred_mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        mask_img = (pred_mask * 255).astype(np.uint8)
        cv2.imwrite(args.output_path, mask_img)
        print(f"Tampering mask saved to: {args.output_path}")

if __name__ == '__main__':
    run_inference()