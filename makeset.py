from moviepy.editor import *
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class VideoDataset(Dataset):
    r"""
        Args:
            root_folder (str): The path of dataset.
            fpath_label (str): The path of label.
            transform (bool): Determines whether to preprocess dataset. Default is False.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 32.
    """

    def __init__(self, root_folder, fpath_label, transform=None, clip_len=32):
        f = open(fpath_label)
        l = f.readlines()
        f.close()
        fpaths = list()
        labels = list()
        for item in l:
            path = item.split()[0]
            label = item.split()[1]
            label = int(label)
            fpaths.append(path)
            labels.append(label)

        self.root_folder = root_folder
        self.fpaths = fpaths
        self.labels = labels
        self.transform = transform
        self.label_size = len(self.labels)
        self.clip_len = clip_len
        self.resize_height = 224
        self.resize_width = 224

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        # Loading and preprocessing.
        frames_dir = self.root_folder + "/" + self.fpaths[index]
        buffer = self.load_frames(frames_dir)
        buffer = self.crop(buffer, self.clip_len)
        label = self.labels[index]

        framesx = torch.tensor(buffer)
        labelx = torch.tensor(label)

        return (framesx, labelx)

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, 3, self.resize_height, self.resize_width), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = Image.open(frame_name).convert("RGB")

            if not self.transform == None:
                frame = self.transform(frame)
                frame = frame.numpy()

            buffer[i] = frame

        # convert from [D, C, H, W] format to [C, D, H, W] (what PyTorch uses)
        buffer = buffer.transpose((1, 0, 2, 3))

        return buffer

    def crop(self, buffer, clip_len):
        if buffer.shape[1] > clip_len:
            # Randomly select start indices in order to crop the video
            time_index = np.random.randint(buffer.shape[1] - clip_len)
        else:
            time_index = 0

        buffer = buffer[:, time_index:time_index + clip_len, :, :]

        return buffer