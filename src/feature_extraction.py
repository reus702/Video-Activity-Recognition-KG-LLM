import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.video import (
    r2plus1d_18,
    R2Plus1D_18_Weights
)


VIDEO_WEIGHTS = R2Plus1D_18_Weights.KINETICS400_V1


def preprocess_frames(frames, transform):
    '''Convert sampled BGR frames to the encoder's RGB input tensor.'''
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    video = torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2)
    return transform(video)


class VideoFeatureExtractor:

    def __init__(self, device=None):
        '''
        Load the pretrained R(2+1)D-18 encoder and freeze its weights.
        '''

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.preprocess = VIDEO_WEIGHTS.transforms()
        self.model = r2plus1d_18(weights=VIDEO_WEIGHTS)

        # Remove the Kinetics-400 classification layer.
        self.model.fc = nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

        # Disable gradient updates for the encoder.
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def extract(self, frames):
        '''
        Convert BGR frames to RGB and return a 512-dimensional NumPy embedding.
        '''

        # B, C, T, H, W
        video = preprocess_frames(frames, self.preprocess).unsqueeze(0).to(self.device)

        embedding = self.model(video)

        return embedding.squeeze(0).cpu().numpy()
