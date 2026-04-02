import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import os

class VideoVAEDataset(Dataset):
    def __init__(self, csv_path, use_text=True, num_frames=16, resolution=256):
        self.data = pd.read_csv(csv_path)  # columns: path, text
        self.use_text = use_text
        self.num_frames = num_frames
        self.resolution = resolution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = row['path']
        text = row['text'] if self.use_text else ''

        frames = self.load_video_frames(video_path)
        if len(frames) < self.num_frames:
            # 循环填充
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
        else:
            start = np.random.randint(0, len(frames) - self.num_frames + 1)
            indices = range(start, start + self.num_frames)

        sampled = [frames[i] for i in indices]
        # 缩放并归一化
        tensor_list = []
        for frame in sampled:
            frame = cv2.resize(frame, (self.resolution, self.resolution))
            frame = frame.astype(np.float32) / 127.5 - 1.0  # [-1,1]
            frame = torch.from_numpy(frame).permute(2,0,1)  # (C,H,W)
            tensor_list.append(frame)
        video_tensor = torch.stack(tensor_list, dim=1)  # (C,T,H,W)

        sample = {'video': video_tensor}
        if self.use_text and text:
            # 简化：实际应使用 Flan-T5 编码文本
            # 这里返回占位符，训练时可替换为真实编码器
            sample['text_embeds'] = torch.randn(1, 768)  # 占位
        return sample

    def load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
