import clip
from datasets import load_dataset
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from ddpm.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# TODO: Add Layer Modulation to Unet
model = Unet(
    dim = 64,
    dim_latent = 64,
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

# dataset = load_dataset('imagenet-1k')
# print(dataset)
def _convert_image_to_rgb(image):
    return image.convert("RGB")
transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
augment = transforms.RandAugment()
dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True)
class SelfSupervisedDataset(Dataset):
    def __init__(self, dataset, transform, augment):
        self.dataset = dataset
        self.transform = transform
        self.augment =augment
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image = self.dataset[index][0]
        aug_img = self.augment(image)
        return self.transform(image), self.transform(aug_img)
    
ssl_dataset = SelfSupervisedDataset(dataset, transform, augment)
# TODO: Add encoder to Trainer
trainer = Trainer(
    diffusion,
    ssl_dataset,
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()