# test.py
import os
import argparse
import glob
import time
import numpy as np
import torch
import rasterio

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from collections import OrderedDict
from model import GeneratorRRDB

LR_MIN = 0.0206
LR_MAX = 0.2737
HR_MIN = 0.0191
HR_MAX = 0.4274

def load_and_preprocess_lr(lr_path, device):
    """
    Loads a TIF image and normalizes it to the [0, 1] range using the exact same statistics as the training dataset.
    """
    with rasterio.open(lr_path) as src:
        lr_image = src.read().astype(np.float32)
        lr_image = (lr_image - LR_MIN) / (LR_MAX - LR_MIN)
        lr_image = np.clip(lr_image, 0.0, 1.0)
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).to(device)
        return lr_tensor

def postprocess_sr(sr_tensor):
    """
    Converts the model's output tensor from the [0, 1] range back to a standard 8-bit image format (0-255) for saving.
    """
    sr_tensor = torch.clamp(sr_tensor, 0, 1)
    sr_image = sr_tensor.squeeze(0).cpu().detach().numpy()
    sr_image = np.transpose(sr_image, (1, 2, 0)) # from (C, H, W) to (H, W, C)
    sr_image = (sr_image * 255.0).astype(np.uint8)
    return sr_image

def load_hr_for_eval(hr_path):
    """
    Loads the ground truth HR image and correctly scales it to the 0-255 uint8 range for fair comparison with the model's output.
    """
    with rasterio.open(hr_path) as src:
        hr_image = src.read().transpose(1, 2, 0).astype(np.float32)
        hr_image_01 = (hr_image - HR_MIN) / (HR_MAX - HR_MIN)
        hr_image_scaled = hr_image_01 * 255.0
        hr_image_uint8 = np.clip(hr_image_scaled, 0, 255).astype(np.uint8)
        return hr_image_uint8

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: GPU ")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU ")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    print("Loading generator model...")
    try:
        model = GeneratorRRDB(channels=3, num_res_blocks=32).to(device)
        print(f"Loading checkpoint: {args.weights_path}")
        state_dict = torch.load(args.weights_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print("Successfully loaded weights from checkpoint.")
        
        model.eval()
    except Exception as e:
        print(f"CRITICAL ERROR during model loading: {e}")
        return

    lr_files = sorted(glob.glob(os.path.join(args.lr_dir, '*.TIF')))
    if not lr_files:
        print(f"Error: No .TIF files found in LR directory: {args.lr_dir}")
        return
    total_psnr, total_ssim, total_time = 0.0, 0.0, 0.0
    image_count = 0

    with torch.no_grad():
        for lr_path in lr_files:
            base_filename = os.path.basename(lr_path)
            hr_path = os.path.join(args.hr_dir, base_filename)
            
            if not os.path.exists(hr_path):
                print(f"Warning: Corresponding HR file for {base_filename} not found. Skipping.")
                continue

            print(f"\nProcessing: {base_filename}")
            
            lr_tensor = load_and_preprocess_lr(lr_path, device)

            start_time = time.time()
            sr_tensor = model(lr_tensor)
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            print(f"  Inference time: {inference_time:.4f} seconds")

            sr_image_uint8 = postprocess_sr(sr_tensor)
            hr_image_uint8 = load_hr_for_eval(hr_path)
            
            if sr_image_uint8.shape != hr_image_uint8.shape:
                print(f"  Warning: Shape mismatch SR {sr_image_uint8.shape} vs HR {hr_image_uint8.shape}. Skipping metrics.")
            else:
                try:
                    psnr_val = psnr(hr_image_uint8, sr_image_uint8, data_range=255)
                    ssim_val = ssim(hr_image_uint8, sr_image_uint8, channel_axis=-1, data_range=255)
                    
                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    image_count += 1
                    
                    print(f"  PSNR: {psnr_val:.4f} dB")
                    print(f"  SSIM: {ssim_val:.4f}")
                except Exception as e:
                    print(f"  Error calculating metrics: {e}")

            try:
                save_path = os.path.join(args.output_dir, f"sr_{base_filename.replace('.TIF', '.png')}")
                Image.fromarray(sr_image_uint8).save(save_path)
                print(f"  Saved SR image to: {save_path}")
            except Exception as e:
                print(f"  Error saving SR image: {e}")

    # --- Calculate and Print Average Metrics ---
    if image_count > 0:
        avg_psnr = total_psnr / image_count
        avg_ssim = total_ssim / image_count
        avg_time = total_time / len(lr_files)
        print("\nEvaluation Summary ")
        print(f"Processed {len(lr_files)} images ({image_count} with valid metrics).")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average Inference Time: {avg_time:.4f} seconds/image")
    else:
        print("\nNo valid images were processed for metrics.")

    print("\nTesting complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained PyTorch Real-ESRGAN generator.")
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
