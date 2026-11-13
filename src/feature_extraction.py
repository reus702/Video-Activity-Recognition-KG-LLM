import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.video import (
    r2plus1d_18,
    R2Plus1D_18_Weights
)

class VideoFeatureExtractor:

    def __init__(self, device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        weights = R2Plus1D_18_Weights.DEFAULT

        self.preprocess = weights.transforms()

        self.model = r2plus1d_18(weights=weights)

        # Rimuove il classificatore Kinetics-400
        self.model.fc = nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

        # Encoder congelato
        self.model.requires_grad_(False)

    '''
    Function used to transform frames in tensors, applies preprocessing 
    phase and passed the clip to R(2+1)D.
    '''
    @torch.inference_mode()
    def extract(self, frames):

        rgb_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            for f in frames
        ]

        # T, H, W, C
        video = np.stack(rgb_frames)

        # T, C, H, W
        video = torch.from_numpy(video).permute(
            0, 3, 1, 2
        )

        # Preprocessing ufficiale dei pesi
        video = self.preprocess(video)

        # B, C, T, H, W
        video = video.unsqueeze(0).to(self.device)

        embedding = self.model(video)

        return embedding.squeeze(0).cpu().numpy()