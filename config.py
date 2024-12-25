import os
import torch

LOAD_MODEL = False
SAVE_MODEL = True
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 16
LAMBDA_GP = 10
LOW_RES = 96
HIGH_RES = LOW_RES * 4

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
INTERMEDIA_SIZE = 96

IMG_CHANNELS = 3
EXP_NAME = f'exp_{LOW_RES}_{INTERMEDIA_SIZE}_{HIGH_RES}'
# EXP_NAME = f'exp_{LOW_RES}_{INTERMEDIA_SIZE}_{HIGH_RES}'
CHECKPOINT_GEN = os.path.join(EXP_NAME, "gen.pth")
CHECKPOINT_DISC = os.path.join(EXP_NAME, "disc.pth")


TRAIN_DATASET_PATH = 'SUNRGBD-train_images_384'
VALID_DATASET_PATH = 'SUNRGBD-valid_images_384'
VALID_SAVE_PATH = os.path.join(EXP_NAME, 'saved')