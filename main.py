import clip
from datasets import load_dataset
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from dataset import AugmentedCompositionDataset
import argparse

from ddpm.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# TODO: Add Layer Modulation to Unet
model = Unet(
    dim = 128,
    dim_latent = 128,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

clip_model, _ = clip.load('RN50')
encoder = clip_model.visual.float()

diffusion = GaussianDiffusion(
    model,
    encoder,
    image_size = 224,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

dataset_path = "/home/ubuntu/22dat.dh/CZSL/mit-states"
#dataset_path = "/home/ubuntu/22hao.vc/diffusion/mit-states"
train_dataset = AugmentedCompositionDataset(dataset_path,
                                    phase='train',
                                    split='compositional-split-natural')

args = argparse.Namespace(
    n_last_blocks = 4,
    avgpool_patchtokens = False,
    arch = 'resnet',
    patch_size = 16,
    pretrained_weights = '',
    epochs = 100,
    batch_size_per_gpu = 128,
    data_path = dataset_path,
    output_dir = ".",
    num_labels = 1000,
    evaluate = True
)

trainer = Trainer(
    diffusion,
    train_dataset,
    args,
    train_batch_size = 4,
    train_lr = 8e-5,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 4,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()