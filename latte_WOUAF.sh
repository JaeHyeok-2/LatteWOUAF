python latte_WOUAF.py \
    --pretrained_model_name_or_path Latte/vae \
    --center_crop --random_flip \
    --dataloader_num_workers 4 \
    --train_steps_per_epoch 1_000 \
    --max_train_steps 50_000 \

