import argparse
import os
import sys
import numpy as np
import torch
import piq

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from model import GeneratorRRDB, UNetDiscriminatorSN, FeatureExtractor
from dataset import Oli2MSIDataset

os.makedirs('images/training', exist_ok=True)
os.makedirs('images/validation', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--lr_dir', type=str, required=True)
parser.add_argument('--hr_dir', type=str, required=True)
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--val_interval', default=1, type=int)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--val_split', default=0.2, type=float)
parser.add_argument('--resume_checkpoint', type=str, default=None)

opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, path='saved_models/generator_best.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_psnr_max = -np.inf 
        self.path = path

    def __call__(self, val_psnr, model):
        score = val_psnr
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_psnr, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_psnr, model)
            self.counter = 0

    def save_checkpoint(self, val_psnr, model):
        if self.verbose:
            print(f'Validation PSNR improved ({self.val_psnr_max:.4f} --> {val_psnr:.4f}). Saving best model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_psnr_max = val_psnr

generator = GeneratorRRDB(channels=3, num_res_blocks=32).to(device)
discriminator = UNetDiscriminatorSN(num_in_ch=3).to(device)
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()

criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.9, 0.999))

start_epoch = 0
if opt.resume_checkpoint:
    if os.path.isfile(opt.resume_checkpoint):
        print(f"Resuming training from checkpoint: {opt.resume_checkpoint}")
        checkpoint = torch.load(opt.resume_checkpoint, map_location=device)
        start_epoch = checkpoint.get('epoch', -1) + 1
        try:
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            print(f"Successfully loaded all states. Resuming from epoch {start_epoch}.")
        except Exception as e:
            print(f"Error loading state dictionaries: {e}")
            sys.exit(1)
    else:
        print(f"Warning: Checkpoint file not found at '{opt.resume_checkpoint}'. Starting from scratch.")


try:
    image_filenames = sorted([f for f in os.listdir(opt.hr_dir) if f.lower().endswith(('.tif', '.tiff'))])
    if not image_filenames:
        print(f"Error: No image files found in {opt.hr_dir}")
        sys.exit(1)
    hr_filepaths = [os.path.join(opt.hr_dir, f) for f in image_filenames]
    lr_filepaths = [os.path.join(opt.lr_dir, f) for f in image_filenames]
except FileNotFoundError:
    print(f"Error: Directory not found: {opt.hr_dir}")
    sys.exit(1)

hr_train_files, hr_val_files, lr_train_files, lr_val_files = train_test_split(
    hr_filepaths, lr_filepaths, test_size=opt.val_split, random_state=42
)
print(f"Found {len(image_filenames)} images.")
print(f"Splitting data: {len(hr_train_files)} for training, {len(hr_val_files)} for validation.")

if not hr_val_files:
    print("Error: Validation set is empty. Adjust --val_split or add more images.")
    sys.exit(1)

train_set = Oli2MSIDataset(lr_files=lr_train_files, hr_files=hr_train_files, is_train=True)
val_set = Oli2MSIDataset(lr_files=lr_val_files, hr_files=hr_val_files, is_train=False)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

def validate_and_save(epoch):
    generator.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for i, (lr_val, hr_val) in enumerate(val_loader):
            lr_val, hr_val = lr_val.to(device), hr_val.to(device)
            sr_val = torch.clamp(generator(lr_val), 0, 1)
            psnr = piq.psnr(sr_val, hr_val, data_range=1.0)
            ssim = piq.ssim(sr_val, hr_val, data_range=1.0)
            psnr_scores.append(psnr.item())
            ssim_scores.append(ssim.item())
            if i < 3: 
                lr_bicubic = torch.nn.functional.interpolate(lr_val, scale_factor=4, mode='bicubic', align_corners=False)
                save_image(torch.cat((lr_bicubic, sr_val, hr_val), -1), f"images/validation/epoch_{epoch}_img_{i}.png", normalize=True)
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    print(f"\n[Validation] PSNR: {avg_psnr:.4f} dB | SSIM: {avg_ssim:.4f}\n")
    return avg_psnr

# Training Loop 
print(" Starting Training...")
early_stopping = EarlyStopping(patience=opt.patience, verbose=True)

for epoch in range(start_epoch, opt.n_epochs):
    generator.train()
    for i, (imgs_lr, imgs_hr) in enumerate(train_loader):
        imgs_lr, imgs_hr = imgs_lr.to(device), imgs_hr.to(device)

        #  Train Generator 
        optimizer_G.zero_grad()
        gen_hr = generator(imgs_lr)
        
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        pred_fake_G = discriminator(gen_hr)
        
        valid = torch.ones(pred_fake_G.shape, requires_grad=False).to(device)
        loss_GAN = criterion_GAN(pred_fake_G, valid)
        
        with torch.no_grad():
            real_features = feature_extractor(imgs_hr)
        gen_features = feature_extractor(gen_hr)
        loss_content = sum(criterion_content(gen_f, real_f) for gen_f, real_f in zip(gen_features, real_features))

        loss_G = loss_content + 1e-3 * loss_GAN + 1e-2 * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        
        pred_real = discriminator(imgs_hr)
        pred_fake_D = discriminator(gen_hr.detach())
        
        valid = torch.ones(pred_real.shape, requires_grad=False).to(device)
        fake = torch.zeros(pred_fake_D.shape, requires_grad=False).to(device)

        loss_real = criterion_GAN(pred_real, valid)
        loss_fake = criterion_GAN(pred_fake_D, fake)
        
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i+1}/{len(train_loader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                  f"[Pixel: {loss_pixel.item():.4f}] [Content: {loss_content.item():.4f}]")


    if (epoch + 1) % opt.val_interval == 0:
        val_psnr = validate_and_save(epoch)
        early_stopping(val_psnr, generator)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    checkpoint_path = 'saved_models/latest_checkpoint.pth'
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)

print("Training complete.")
