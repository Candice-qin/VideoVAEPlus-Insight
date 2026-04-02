import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml
import os

from video_vae_plus import VideoVAEPlus, VideoVAELoss
from dataset import VideoVAEDataset

class VideoVAETrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VideoVAEPlus(
            latent_channels=self.cfg['model']['latent_channels'],
            use_text_guidance=self.cfg['model']['use_text_guidance']
        ).to(self.device)

        self.criterion = VideoVAELoss(
            l1_weight=self.cfg['loss']['l1_weight'],
            lpips_weight=self.cfg['loss']['lpips_weight'],
            temporal_weight=self.cfg['loss']['temporal_weight'],
            motion_weight=self.cfg['loss']['motion_weight'],
            kl_weight=self.cfg['loss']['kl_weight']
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg['training']['lr'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg['training']['epochs'])

        self.train_loader = DataLoader(
            VideoVAEDataset(self.cfg['data']['train_dir'], use_text=self.cfg['model']['use_text_guidance']),
            batch_size=self.cfg['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DataLoader(
            VideoVAEDataset(self.cfg['data']['val_dir'], use_text=self.cfg['model']['use_text_guidance']),
            batch_size=self.cfg['training']['batch_size'],
            shuffle=False
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for step, batch in enumerate(pbar):
            video = batch['video'].to(self.device)
            text = batch.get('text_embeds', None)
            if text is not None:
                text = text.to(self.device)

            self.optimizer.zero_grad()
            recon, mean, logvar = self.model(video, text_embeds=text)
            global_step = epoch * len(self.train_loader) + step
            loss = self.criterion(recon, video, mean, logvar, step=global_step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                video = batch['video'].to(self.device)
                text = batch.get('text_embeds', None)
                if text is not None:
                    text = text.to(self.device)
                recon, mean, logvar = self.model(video, text_embeds=text)
                loss = self.criterion(recon, video, mean, logvar)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.cfg['training']['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, f'checkpoints/epoch_{epoch}.pt')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    trainer = VideoVAETrainer(sys.argv[1])
    trainer.train()
