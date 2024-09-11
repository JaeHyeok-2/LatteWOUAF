#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Image
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler

from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import itertools
from attribution import MappingNetwork
from customization import customize_vae_decoder
import inspect
from torchvision.utils import save_image
import lpips
import wandb
from attack_methods.attack_initializer import attack_initializer #For augmentation
import hydra
from hydra import compose, initialize
import time 
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt 

logger = get_logger(__name__, log_level="INFO")


from utils import *

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import cv2
import torch.nn.functional as F
import lpips

import matplotlib.pyplot as plt
import numpy as np


def save_images_as_grid(original_images, generated_images, file_name, nrows=2, ncols=4, group_gap=0.05):
    fig = plt.figure(figsize=(12, 8))
    
    total_height = 1.0
    height_per_group = (total_height - group_gap) / 2
    
    # Plot Original Images
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = fig.add_axes([j / ncols, 0.5 + i * height_per_group / nrows, 1 / ncols, height_per_group / nrows])
            if idx < len(original_images):
                img = original_images[idx].cpu().numpy().transpose(1, 2, 0)
                img = np.clip((img * 0.5) + 0.5, 0, 1)  # Normalize and clip
                ax.imshow(img)
            ax.axis('off')

    # Plot Generated Images
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = fig.add_axes([j / ncols, i * height_per_group / nrows, 1 / ncols, height_per_group / nrows])
            if idx < len(generated_images):
                img = generated_images[idx].cpu().numpy().transpose(1, 2, 0)
                img = np.clip((img * 0.5) + 0.5, 0, 1)  # Normalize and clip
                ax.imshow(img)
            ax.axis('off')

    # Add titles above each group
    fig.text(0.5, 0.98, 'Original Images', ha='center', fontsize=16, weight='bold')
    fig.text(0.5, 0.48, 'Generated Images', ha='center', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--just_val",
        default=False,
        help= "Whether just val or not "
    )
    parser.add_argument(
        "--quiet",
        type=bool,
        default=False,
        help= "Whether show Nan-Gradient"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_1",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--lr_mult",
        type=float,
        default=1,
        help="Learning rate multiplier for the affine layers",
    )
    parser.add_argument(
        "--pre_latents",
        type=str,
        default=None,
        help="Path to pre-extracted latents for validation",
    )
    parser.add_argument(
        "--phi_dimension",
        type=int,
        default=48,
        help="phi_dimension",
    )
    parser.add_argument(
        "--int_dimension",
        type=int,
        default=64,
        help="intermediate dimension",
    )
    parser.add_argument(
        "--mapping_layer",
        type=int,
        default=2,
        help="FC layers of mapping network",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default='',
        help=(
            "which attack methods to apply ('c' | 'r' | 'g' | 'b' | 'n' | 'e' | 'j' | ... | 'crgbnej' or 'all' || 'AE_b_1' | 'AE_c_6' | ...)"
            "e.g. 'cr' denotes random cropping ('c') and rotation ('r')"
            "Use 'crgbnej' or 'all' for the combined attack in the paper"
        ),
    )
    parser.add_argument(
        "--num_gradient_from_last",
        type=int,
        default=1,
        help="number of getting gradient from last in the denoising loop",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Latte/Latte-1",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/storage3/youngdong/dataset/webvid",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="프레임 개수만큼"
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        default=1000,
        help="Number of training steps per epoch. If provided, limits the number of iterations for each epoch",
    )
    parser.add_argument("--num_train_epochs", type=int, default=4000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--cosine_cycle",
        type=int,
        default=1000,
        help=(
            "cosine_with_restarts option for cycle"
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    ),

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args
        

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def get_phis(phi_dimension, batch_size ,eps = 1e-8, seed=42):
    torch.manual_seed(seed)
    phi_length = phi_dimension
    b = batch_size
    # phi = torch.empty(b,phi_length).uniform_(0,1)
    phi = torch.empty(1, phi_length).uniform_(0,1).repeat(b,1)
    return torch.bernoulli(phi) 


def check_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def prepare_extra_step_kwargs(generator, eta, scheduler):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs



def decode_latents(vae, latents, enconded_fingerprint):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents, enconded_fingerprint).sample
    image = image.clamp(-1,1)
    return image


def get_params_optimize(vaed, mapping_network, decoding_network):
    params_to_optimize = itertools.chain(vaed.parameters(), mapping_network.parameters(), decoding_network.parameters())
    return params_to_optimize


def acc_calculation(args, phis, decoding_network, generated_image, bsz = None, vae = None):
    reconstructed_keys = decoding_network(generated_image)
    gt_phi = (phis > 0.5).int()
    reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
    bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension

    return bit_acc


def load_val_latents(args, batch_size, val_step):
    val_latents = None
    step = val_step*args.train_batch_size
    for i in range(batch_size):
        vl = torch.load(os.path.join(args.pre_latents, f'{step+i}.pth'))
        if val_latents is None:
            val_latents = vl.unsqueeze(0)
        else:
            val_latents = torch.cat((val_latents, vl.unsqueeze(0)), 0)

    return val_latents




#        val(vae, map_net, dec_net, val_file)
def val(args, vae, mapping_network, decoding_network, val_file, test_transforms, weight_dtype, valid_aug=None, resize=None, metrics=None):

    vae.decoder.load_state_dict(torch.load("/home/jh/jh/WOUAF_0240/output/phi_48/vae_decoder_48_8_14.pth"))
    # mapping_network.load_state_dict(torch.load("/home/jh/jh/WOUAF_0240/output/phi_48/mapping_network_48_8_14.pth"))
    decoding_network.load_state_dict(torch.load("/home/jh/jh/WOUAF_0240/output/phi_48/decoding_network_48_8_14.pth"))


    list_val_bit_acc = []

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)
    #Change network eval mode
    vae.eval()
    mapping_network.eval()
    decoding_network.eval()
    device = "cuda"
    count = 0
    total_loss = 0. 
    val_loss =0.
    loss_lpips = 0.

    csv_file = 'latte_web2video_test/validation_results.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Loss', 'Average Bit Accuracy', "LPIPS"])  # 헤더 작성

    with torch.no_grad():
        for (batch_idx, video_file) in tqdm(enumerate(val_file),  desc="Current Video "):        
            # torch.manual_seed(42 + batch_idx) # video 마다 서로다른 Watermarking Bits 를 가져야 하므로 seed값 변경 

            video_path = os.path.join(os.path.join(args.data_path,'val'), video_file) 

            cap = cv2.VideoCapture(video_path) 
            frame_tensors = []

            # print(f"Processing Video :{video_file}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame= max(0, total_frames // 2 -8)
            end_frame = min(total_frames, start_frame+ 8)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret: 
                    break 

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = test_transforms(frame)  # [-1, 1]


                frame_tensors.append(frame) 

            cap.release() 

            # 첫번째 비디오 읽어옴.         
            frame_tensors =  torch.stack(frame_tensors)
            # print(frame_tensors.shape) 
            val_dataset = TensorDataset(frame_tensors) 
            val_dataloader = DataLoader(val_dataset, batch_size = args.train_batch_size, shuffle=False)
            
            num_batches = len(val_dataloader)

            phis = get_phis(args.phi_dimension, args.train_batch_size).to(device)

            for image_tensor in val_dataloader :

                original_image = image_tensor[0].to(accelerator.device) # [-1, 1]
        
                image_tensor = image_tensor[0]

                latents = vae.encode(image_tensor.to(weight_dtype).to(accelerator.device)).latent_dist.sample()

                latents = latents *0.18215 

                encoded_phis=mapping_network(phis)
                encoded_phis = encoded_phis.to(weight_dtype)

                generated_image = decode_latents(vae, latents, encoded_phis)   # [-1, 1]
                


                if 'AE_' in args.attack:
                    augmented_image = valid_aug.forward((generated_image / 2 + 0.5).clamp(0, 1))['x_hat'].clamp(0, 1)
                else:
                    augmented_image = valid_aug((generated_image / 2 + 0.5).clamp(0, 1))


                reconstructed_keys = decoding_network(augmented_image)


                # Proceed with the rest of the code only if no NaN or Inf values were found
                loss_key = F.binary_cross_entropy_with_logits(reconstructed_keys, phis)
                loss_lpips_reg = loss_fn_vgg.forward(generated_image, resize(original_image)).mean()
                loss = loss_key + loss_lpips_reg


                gt_phi = (phis > 0.5).int()
                reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
                bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension
                list_val_bit_acc.append(bit_acc.mean().item())

                avg_loss = loss.mean()

                val_loss += avg_loss 
                loss_lpips += loss_lpips_reg
                print(f"loss_key:" ,loss_key.detach().item(), "loss_lpips:",loss_lpips_reg.item(), "bit_acc =", bit_acc)
                logs = {"loss_key": loss_key.detach().item(), "loss_lpips":loss_lpips_reg.item() }

                save_images_as_grid(original_image, generated_image, f"latte_web2video_test/video_{batch_idx}.png")
    loss_lpips /= len(val_file)
    total_loss = val_loss / len(val_file)

    average_bit_acc = sum(list_val_bit_acc) / len(list_val_bit_acc)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([total_loss.item(), average_bit_acc, loss_lpips.item()])

    print(f"Validation results saved to {csv_file}")


########################################################################################################333
########################################################################################################333
########################################################################################################333


def main():
    args = parse_args()

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        cfg_hydra = compose(config_name="metrics", overrides=[])

    metrics = []
    for _, cb_conf in cfg_hydra.items():
        metrics.append(hydra.utils.instantiate(cb_conf))



    global accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=logging_dir,
    )



    print("Accelerator is local main procss : ", accelerator.is_local_main_process) 

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)


    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),

            transforms.Normalize([0.5], [0.5]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    optimizer_cls = torch.optim.AdamW
    vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    mapping_network = MappingNetwork(args.phi_dimension, args.int_dimension, num_layers=args.mapping_layer)
    decoding_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    decoding_network.fc = torch.nn.Linear(2048, args.phi_dimension)
    print("Loading Network")


    loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)

    # Weight modulation to vae's decoder
    vae = customize_vae_decoder(vae, args.int_dimension, args.lr_mult)

    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    vae.decoder.mid_block.attentions[0].to_v.bias.requires_grad = False

    print("load Fingerprint VAE")
    # For mixed precision training we cast the vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Move text_encode and vae to gpu and cast to weight_dtypefd
    vae.to(accelerator.device, dtype=weight_dtype)
    mapping_network.to(accelerator.device, dtype=weight_dtype)
    decoding_network.to(accelerator.device, dtype=weight_dtype)


    optimizer = optimizer_cls(
        get_params_optimize(vae.decoder, mapping_network, decoding_network),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    print(f"Optimizer 정보 : lr = {args.learning_rate}, betas = {args.adam_beta1, args.adam_beta2}, weight_decay = {args.adam_weight_decay}, eps = {args.adam_epsilon}")


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles = args.cosine_cycle * args.gradient_accumulation_steps,
    )


    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))




    #Augmentation
    train_aug = attack_initializer(args, is_train = True, device = accelerator.device)
    valid_aug = attack_initializer(args, is_train = False, device = accelerator.device)

    if args.resolution == 256:
        resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    else:
        resize = torch.nn.Identity()

    # Setup all metrics
    for metric in metrics:
        metric.setup(accelerator, args)


    device = torch.device("cuda")


    
    local_step = 0
    train_loss = 0.0
    list_train_bit_acc = []
    
    num_epochs = args.num_train_epochs





    train_path =  os.path.join(args.data_path,'train') 
    train_video_files = [f for f in os.listdir(train_path) if f.endswith('mp4')]
    training_len = [f for f in os.listdir()]
    training_video_files = train_video_files

    val_path = os.path.join(args.data_path, 'val')
    val_video_files = [f for f in os.listdir(val_path) if f.endswith('mp4')] 

    print("Number of dataset :", len(training_video_files), len(val_video_files))
    # val_video_files = video_files[training_len:]


    # if args.just_val : 
    #     val(args, vae, mapping_network, decoding_network, val_video_files, test_transforms, weight_dtype, valid_aug, resize=resize)
    #     exit(-1)


    torch.manual_seed(42) 
    # phis = get_phis(args.phi_dimension, args.train_batch_size).to(device)
    phis = torch.tensor([[1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.]]).to(device)
    for epoch in tqdm(range(num_epochs),desc = "NUM EPOCHS"):
        vae.train()
        mapping_network.train()
        decoding_network.train()

                

        start_time = time.time()

            # video folder를 다 읽어오자. 
        for (batch_idx, video_file) in tqdm(enumerate(training_video_files),  desc="Current Video "):        
            # torch.manual_seed(42 + batch_idx) # video 마다 서로다른 Watermarking Bits 를 가져야 하므로 seed값 변경 

            video_path = os.path.join(os.path.join(args.data_path,'train'), video_file) 

            cap = cv2.VideoCapture(video_path) 
            frame_tensors = []

            # print(f"Processing Video :{video_file}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame= max(0, total_frames // 2 -8)
            end_frame = min(total_frames, start_frame+ 8)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret: 
                    break 

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = train_transforms(frame)  # [-1, 1]


                frame_tensors.append(frame) 

            cap.release() 

            # 첫번째 비디오 읽어옴.         
            frame_tensors =  torch.stack(frame_tensors)
            # print(frame_tensors.shape) 
            train_dataset = TensorDataset(frame_tensors) 
            train_dataloader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle=False)
            
            num_batches = len(train_dataloader)

            # phis = get_phis(args.phi_dimension, args.train_batch_size).to(device)

            for image_tensor in train_dataloader :

                # print(image_tensor[0])
                original_image = image_tensor[0].to(accelerator.device) # [-1, 1]
                
                image_tensor = image_tensor[0]


                # latents = accelerator.unwrap_model(vae).encode(image_tensor.to(weight_dtype).to(accelerator.device)).latent_dist.sample()
                
                latents = vae.encode(image_tensor.to(weight_dtype).to(accelerator.device)).latent_dist.sample()
                # print(latents[0])
                latents = latents *0.18215 

                encoded_phis=mapping_network(phis)
                encoded_phis = encoded_phis.to(weight_dtype)
                # print(encoded_phis.dtype)
                
                generated_image = decode_latents(vae, latents, encoded_phis)   # [-1, 1]
                
                # print(latents.dtype, original_image.dtype, phis.dtype, encoded_phis.dtype, generated_image.dtype)

                if 'AE_' in args.attack:
                    augmented_image = train_aug.forward((generated_image / 2 + 0.5).clamp(0, 1))['x_hat'].clamp(0, 1)
                else:
                    augmented_image = train_aug((generated_image / 2 + 0.5).clamp(0, 1))
                    # augmented_image= generated_image
 

                reconstructed_keys = decoding_network(augmented_image)
                # print(original_image.shape, generated_image.shape)
                # print(phis)

                # Proceed with the rest of the code only if no NaN or Inf values were found


                if torch.isnan(reconstructed_keys).any():
                    print("Reconstructed Key Contains NaN Values!!")
                if torch.isinf(reconstructed_keys).any():
                    print("Reconstructed Key Contains Inf Values!!")

                if torch.isnan(encoded_phis).any():
                    print("Encoded Phis contains NaN Values!!")
                if torch.isinf(encoded_phis).any():
                    print("Encoded Phis contains Inf Values!!")

                if torch.isnan(latents).any():
                    print("Latents contains NaN Values!!")
                if torch.isinf(latents).any():
                    print("Latents contains Inf Values!!")

                loss_key = F.binary_cross_entropy_with_logits(reconstructed_keys, phis)

                print(f"generated_image range: {generated_image.min().item()} to {generated_image.max().item()}")
                print(f"original_image range: {resize(original_image).min().item()} to {resize(original_image).max().item()}")

                
                loss_lpips_reg = loss_fn_vgg.forward(generated_image, resize(original_image)).mean()
                loss = loss_key + loss_lpips_reg


                gt_phi = (phis > 0.5).int()
                reconstructed_keys = (torch.sigmoid(reconstructed_keys) > 0.5).int()
                bit_acc = ((gt_phi == reconstructed_keys).sum(dim=1)) / args.phi_dimension
                list_train_bit_acc.append(bit_acc.mean().item())

                # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_loss = loss.mean()
                # train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_loss += avg_loss / args.gradient_accumulation_steps
                print(f"loss_key:" ,loss_key.detach().item(), "loss_lpips:",loss_lpips_reg.item(), "bit_acc =", bit_acc)
                logs = {"loss_key": loss_key.detach().item(), "loss_lpips":loss_lpips_reg.item() }
                # print(f"original : {original_image.dtype}, generated : {generated_image.dtype}, Message : {phis.dtype}, retreival Message :{reconstructed_keys.dtype}, ")

                if (batch_idx + 1) % num_batches == 0:
                    # torch.autograd.set_detect_anomaly(True)
                    optimizer.zero_grad()

                    train_loss /= num_batches 
                    train_loss.backward()
                    # print(train_loss)
                    # if accelerator.sync_gradients:s
                    # accelerator.clip_grad_norm_(get_psarams_optimize(accelerator.unwrap_model(vae).decoder, mapping_network, decoding_network), args.max_grad_norm)
                    
                    torch.nn.utils.clip_grad_norm_(
                    get_params_optimize(vae.decoder, mapping_network, decoding_network),
                    args.max_grad_norm 
                   )
                    optimizer.step()
                    lr_scheduler.step()

                    # VAE Decoder의 그라디언트 확인
                    for name, param in vae.decoder.named_parameters():
                        if param.requires_grad and param.grad is None:
                            print(f"Parameter: {name} has no gradient but requires_grad=True")
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"Parameter: {name} has NaN gradients!")
                            # print(f"{name} gradient min: {param.grad.min()}, max: {param.grad.max()}")

                    # Mapping Network의 그라디언트 확인
                    for name, param in mapping_network.named_parameters():
                        if param.requires_grad and param.grad is None:
                            print(f"Parameter: {name} has no gradient but requires_grad=True")
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"Parameter: {name} has NaN gradients!")
                            # print(f"{name} gradient min: {param.grad.min()}, max: {param.grad.max()}")

                    # Decoding Network의 그라디언트 확인
                    for name, param in decoding_network.named_parameters():
                        if param.requires_grad and param.grad is None:
                            print(f"Parameter: {name} has no gradient but requires_grad=True")
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"Parameter: {name} has NaN gradients!")
                            # print(f"{name} gradient min: {param.grad.min()}, max: {param.grad.max()}")



                    train_loss = 0.0

            print('save pth file....')
            torch.save(vae.decoder.state_dict(), args.output_dir + f"/vae_decoder_{args.phi_dimension}_{args.train_batch_size}_{epoch}.pth")
            torch.save(mapping_network.state_dict(), args.output_dir + f"/mapping_network_{args.phi_dimension}_{args.train_batch_size}_{epoch}.pth")
            torch.save(decoding_network.state_dict(), args.output_dir + f"/decoding_network_{args.phi_dimension}_{args.train_batch_size}_{epoch}.pth")
        end_time = time.time()

        print(f"{epoch} EPOCH당 걸리는 시간!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! :{start_time - end_time}")

    # val(args, accelerator, weight_dtype, generation_scheduler, vae, mapping_network, decoding_network, val_loader, valid_aug, resize, metrics)

    # accelerator.wait_for_everyone()



if __name__ == "__main__":
    main()
