import os
import torch
import argparse
import torchvision


from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_latte import LattePipeline
from models import get_models
from utils import save_video_grid
import imageio
from torchvision.utils import save_image

import json 

from diffusers.utils.torch_utils import randn_tensor 
from torchvision.models import resnet50, ResNet50_Weights 
from attribution import MappingNetwork 
from customization import customize_vae_decoder
import numpy as np 
import torchvision.utils as vutils 
import matplotlib as plt 
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import csv 
import random

# 텐서를 이미지로 변환하는 함수
def save_images_from_tensor_list(tensor_list, folder_name, prefix=None):
    # PIL 이미지 변환기 초기화
    to_pil = ToPILImage()
    
    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 각 텐서를 순차적으로 PIL 이미지로 변환 후 저장
    for i, tensor in enumerate(tensor_list):
        # 텐서를 PIL 이미지로 변환
        image = to_pil(tensor)
        # 이미지 경로 생성 및 저장
        image_path = os.path.join(folder_name, f"frame_{i+1}.png")
        image.save(image_path)

def seed_everything_(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def combine_images_in_grid(images_list, grid_size=(2, 4)):
    """

    
    Parameters:
    images_list (list of torch.Tensor): 각 이미지가 [channels, height, width] 형태의 텐서로 구성된 리스트.
    grid_size (tuple): (행의 수, 열의 수)로 그리드의 크기를 정의.
    
    Returns:
    torch.Tensor: 결합된 이미지 텐서.
    """
    num_images = len(images_list)
    print(num_images)
    if num_images != grid_size[0] * grid_size[1]:
        raise ValueError(f"Number of images ({num_images}) does not match the grid size {grid_size[0]}x{grid_size[1]}")
    print(images_list[0].shape)
    images_list = [image.squeeze(0) for image in images_list]
    channels, height, width = images_list[0].shape
    grid_h, grid_w = grid_size

    # 큰 빈 텐서 생성
    combined_image = torch.zeros((channels, grid_h * height, grid_w * width), dtype=images_list[0].dtype)
    
    for idx, image in enumerate(images_list):
        row = idx // grid_w
        col = idx % grid_w
        combined_image[:, row * height:(row + 1) * height, col * width:(col + 1) * width] = image
    
    return combined_image

# 파라미터 개수와 모델 크기 계산 함수
def count_parameters_and_size(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 메모리 크기 계산 (각 파라미터의 크기는 p.element_size() * p.numel())
    model_size_in_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    model_size_in_megabytes = model_size_in_bytes / (1024 * 1024)  # MB로 변환
    
    return num_params, model_size_in_megabytes

def mapping_weights_name(state_dict):
    key_mapping = {
    "mid_block.attentions.0.query.weight": "mid_block.attentions.0.to_q.weight",
    "mid_block.attentions.0.query.bias": "mid_block.attentions.0.to_q.bias",
    "mid_block.attentions.0.key.weight": "mid_block.attentions.0.to_k.weight",
    "mid_block.attentions.0.key.bias": "mid_block.attentions.0.to_k.bias",
    "mid_block.attentions.0.value.weight": "mid_block.attentions.0.to_v.weight",
    "mid_block.attentions.0.value.bias": "mid_block.attentions.0.to_v.bias",
    "mid_block.attentions.0.proj_attn.weight": "mid_block.attentions.0.to_out.0.weight",
    "mid_block.attentions.0.proj_attn.bias": "mid_block.attentions.0.to_out.0.bias",
    }

    new_state_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict.pop(old_key)

    new_state_dict.update(state_dict)

    return new_state_dict

def load_wouaf_pretrained(args):

    # pretrained_path = [os.path.join(args.wouaf_pretrain_path, path) for path in os.listdir(args.wouaf_pretrain_path)]
    print(args.pretrained_model_path)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", revision= False)
    mapping_network = MappingNetwork(args.wouaf_phi_dimension, args.wouaf_int_dimension, num_layers=args.wouaf_mapping_layer)
    decoding_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    decoding_network.fc = torch.nn.Linear(2048, args.wouaf_phi_dimension)
    
    vae = customize_vae_decoder(vae, args.wouaf_int_dimension, args.wouaf_lr_mult)

    vae_state_dict = mapping_weights_name(torch.load(os.path.join(args.wouaf_pretrain_path,f'vae_decoder_{args.wouaf_phi_dimension}_8_16.pth')))

    params_before_loading, size_before_loading = count_parameters_and_size(vae)
    print(f"Parameters before loading: {params_before_loading}")
    print(f"Model size before loading: {size_before_loading:.2f} MB")
    
    params_before_loading, size_before_loading = count_parameters_and_size(mapping_network)
    print(f"Parameters before loading: {params_before_loading}")
    print(f"Model size before loading: {size_before_loading:.2f} MB")

    params_before_loading, size_before_loading = count_parameters_and_size(decoding_network)
    print(f"Parameters before loading: {params_before_loading}")
    print(f"Model size before loading: {size_before_loading:.2f} MB")


    vae.decoder.load_state_dict(vae_state_dict)
    mapping_network.load_state_dict(torch.load(os.path.join(args.wouaf_pretrain_path,f'mapping_network_{args.wouaf_phi_dimension}_8_16.pth')))
    decoding_network.load_state_dict(torch.load(os.path.join(args.wouaf_pretrain_path,f'decoding_network_{args.wouaf_phi_dimension}_8_16.pth')))
    params_after_loading, size_after_loading = count_parameters_and_size(vae)
    print(f"Parameters after loading: {params_after_loading}")
    print(f"Model size after loading: {size_after_loading:.2f} MB")
    exit(-1)
    return vae, mapping_network, decoding_network 


def get_phis(phi_dimension, batch_size ,eps = 1e-8):
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b,phi_length).uniform_(0,1)
    return torch.bernoulli(phi) + eps


def main(args):
    seed_everything_(123)
    wouaf_parser = argparse.ArgumentParser()
    wouaf_parser.add_argument("--wouaf_pretrain_path", type=str, default='/home/jh/jh/WOUAF_0240/output/phi_48')
    wouaf_parser.add_argument('--wouaf_phi_dimension', type=int, default=48)
    wouaf_parser.add_argument('--wouaf_int_dimension', type=int, default=64) 
    wouaf_parser.add_argument('--wouaf_lr_mult', type=float, default=1)
    wouaf_parser.add_argument('--wouaf_mapping_layer', type=int, default=2)
    wouaf_parser.add_argument("--config", type=str, default="./configs/wbv10m_train.yaml")
    # Sd-2의 vae에 Latte VAE를 그냥 넣어놨음.
    wouaf_parser.add_argument('--pretrained_model_path', default='/home/jh/jh/WOUAF_0240/Latte/Latte-1')
    wouaf_args = wouaf_parser.parse_args()

    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer_model = get_models(args).to(device, dtype=torch.float16)
    
    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float32).to(device)
    else:
        # vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float32).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type,
                                                  clip_sample=False)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                        subfolder="scheduler",
                                                        beta_start=args.beta_start, 
                                                        beta_end=args.beta_end, 
                                                        beta_schedule=args.beta_schedule,
                                                        variance_type=args.variance_type)
    elif args.sample_method == 'DDPM':
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type,
                                                  clip_sample=False)
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'HeunDiscrete':
        scheduler = HeunDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    elif args.sample_method == 'KDPM2AncestralDiscrete':
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type)
    # print(args)
    wouaf_vae_decoder, mapping_network, decoding_network =load_wouaf_pretrained(wouaf_args)
    # vae, mapping_network, decoding_network =load_wouaf_pretrained(wouaf_args)
    


    videogen_pipeline = LattePipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler, 
                                 transformer=transformer_model).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    # vae.to(device, dtype=torch.float16).eval()
    wouaf_vae_decoder.to(device, dtype=torch.float16).eval()
    mapping_network.to(device, dtype=torch.float16).eval()
    decoding_network.to(device, dtype=torch.float16).eval()
    vae.to(device, dtype=torch.float16).eval()

    csv_file_name = f"video_accuracy_results_{wouaf_args.wouaf_phi_dimension}.csv" 
    csv_data = [["Prompt", "Current Video Accuracy"]]

    # torch.manual_seed(123)
    # np.random.seed(123) 
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(123)
    #     torch.cuda.manual_seed_all(123)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



    # phis = get_phis(wouaf_args.wouaf_phi_dimension, args.video_length).to(device)
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
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
         0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1.]]).to(device)
    # print(phis)
    for num_prompt, prompt in enumerate(args.text_prompt):
        print('Processing the ({}) prompt'.format(prompt))
        # videos = videogen_pipeline(prompt, 
        #                         video_length=args.video_length, 
        #                         height=args.image_size[0], 
        #                         width=args.image_size[1], 
        #                         num_inference_steps=args.num_sampling_steps,
        #                         guidance_scale=args.guidance_scale,
        #                         enable_temporal_attentions=args.enable_temporal_attentions,
        #                         num_images_per_prompt=1,
        #                         mask_feature=True,
        #                         enable_vae_temporal_decoder=args.enable_vae_temporal_decoder
        #                         ).video

        result_folder_name = './config/results'
        if not os.path.exists(result_folder_name):
            os.makedirs(result_folder_name, exist_ok=True) 

        if not os.path.exists(os.path.join(result_folder_name, prompt)):
            os.makedirs(os.path.join(result_folder_name, prompt.replace(' ', '_')[:-1]), exist_ok=True) 
        
        latents = videogen_pipeline(
        prompt=prompt, 
        video_length=args.video_length, 
        height=args.image_size[0], 
        width=args.image_size[1], 
        num_inference_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale,
        enable_temporal_attentions=args.enable_temporal_attentions,
        num_images_per_prompt=1,
        mask_feature=True,
        enable_vae_temporal_decoder=args.enable_vae_temporal_decoder,
          # latents로 출력
        output_type='latent_z'
        )

        # print(latents.shape)# [batch, channel, frame, hieght, width] 순으로 뽑히니, -> [batch, frame, channel, height, width]
        latents = latents.permute(0,2,1,3,4)
        print(latents.shape)

        generated_images_list = []
        original_images_list = []
        
        current_video_accuracy = 0. # 현재 비디오의 Bit Accuracy
        for frame_latent in latents :   
            print(frame_latent.shape)
            frame_latent = frame_latent.to(device)

            
            # for frame_latent in latent: 
            # frame_latent.unsqueeze_(0)
            frame_latent = frame_latent * 1/ 0.18215


            encoded_phis = mapping_network(phis).to(device, dtype=torch.float16)
            
            print(frame_latent.dtype, encoded_phis.dtype)
            generated_image = wouaf_vae_decoder.decode(frame_latent, encoded_phis).sample

            # GT Images 
            original_image = vae.decode(frame_latent).sample
            original_image = (original_image /2.0 + 0.5).clamp(0, 1)
            original_images_list.append(original_image)

            # Retrieval Message 
            retrieval_message_logits = decoding_network(generated_image)
            retrieval_message = (torch.sigmoid(retrieval_message_logits) > 0.5).int() 

            generated_image = (generated_image / 2.0 + 0.5).clamp(0, 1)
            generated_images_list.append(generated_image)
            
            bit_accuracy = ((phis == retrieval_message).sum(dim=1)) / wouaf_args.wouaf_phi_dimension 
            current_video_accuracy += bit_accuracy
        
        current_video_accuracy /= args.video_length 

        print(f"PROMPT : {prompt}, Bit Accuracy : {bit_accuracy}")
        csv_data.append([prompt, current_video_accuracy])

        
        # print(g)


        # generated_images_list = combine_images_in_grid(generated_images_list[0])
        # original_images_list =combine_images_in_grid(original_images_list[0])

        print(prompt.replace(' ', '_'))
        # save_image(generated_images_list[0],  f"{os.path.join(result_folder_name, prompt.replace(' ', '_')[:-1])}" + "/generated.png")
        # save_image(original_images_list[0], f"{os.path.join(result_folder_name, prompt.replace(' ', '_')[:-1])}" + "/original.png")
        result_folder_path = os.path.join(result_folder_name, prompt.replace(' ', '_')[:-1])

        save_images_from_tensor_list(generated_images_list[0], os.path.join(result_folder_path,'generated'))
        save_images_from_tensor_list(original_images_list[0], os.path.join(result_folder_path, 'ori'))

    with open(csv_file_name, 'w', newline='') as file : 
        writer = csv.writer(file) 
        writer.writerows(csv_data) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")

    args = parser.parse_args()

    main(OmegaConf.load(args.config))

