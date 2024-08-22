CUDA_VISIBLE_DEVICES=0 python trainval_WOUAF.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-base \
    --dataset_name HuggingFaceM4/COCO \
    --phi_dimension 32 \
    --dataset_config_name 2014_captions --caption_column sentences_raw \
    --center_crop --random_flip \
    --dataloader_num_workers 4 \
    --train_steps_per_epoch 1_000\
    --max_train_steps 50_000 \
    --pre_latents latents/HuggingFaceM4/COCO



# CUDA_VISIBLE_DEVICES=0 python trainval_WOUAF.py \
#     --pretrained_model_name_or_path stabilityai/stable-diffusion-2-base \
#     --dataset_name HuggingFaceM4/COCO \
#     --phi_dimension=48 \
#     --dataset_config_name 2014_captions --caption_column sentences_raw \
#     --center_crop --random_flip \
#     --dataloader_num_workers 4 \
#     --train_steps_per_epoch 1_000 \
#     --max_train_steps 50_000 \
#     --pre_latents latents/HuggingFaceM4/COCO


#     CUDA_VISIBLE_DEVICES=0 python trainval_WOUAF.py \
#     --pretrained_model_name_or_path stabilityai/stable-diffusion-2-base \
#     --dataset_name HuggingFaceM4/COCO \
#     --phi_dimension=64 \
#     --dataset_config_name 2014_captions --caption_column sentences_raw \
#     --center_crop --random_flip \
#     --dataloader_num_workers 4 \
#     --train_steps_per_epoch 1_000 \
#     --max_train_steps 50_000 \
#     --pre_latents latents/HuggingFaceM4/COCO


#     CUDA_VISIBLE_DEVICES=0 python trainval_WOUAF.py \
#     --pretrained_model_name_or_path stabilityai/stable-diffusion-2-base \
#     --dataset_name HuggingFaceM4/COCO \
#     --phi_dimension=128 \
#     --dataset_config_name 2014_captions --caption_column sentences_raw \
#     --center_crop --random_flip \
#     --dataloader_num_workers 4 \
#     --train_steps_per_epoch 1_000 \
#     --max_train_steps 50_000 \
#     --pre_latents latents/HuggingFaceM4/COCO