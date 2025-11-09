accelerate launch train.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --num_train_epochs=600 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=1 \
    --learning_rate=5e-5 \
    --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --num_frames=10\
    --num_workers=4 \
    --enable_xformers_memory_efficient_attention \
    --resume_from_checkpoint="latest" \
    --train_data_path='/u/nthadishetty/data' \
    --output_dir="/u/nthadishetty/VDM_EVFI/train_streo" 
    #--train_data_path="/u/nthadishetty/bs_ergb_data/processed_data/99_BSERGB_MStack/3_TRAINING" \
    #--output_dir="/u/nthadishetty/VDM_EVFI/train_bsergb" 



    # --train_data_path='/u/nthadishetty/data' \
    # --output_dir="/u/nthadishetty/VDM_EVFI/train_streo" 



#    --train_data_path="/u/nthadishetty/bs_ergb_data/processed_data/99_BSERGB_MStack/3_TRAINING" \
#     --output_dir="/u/nthadishetty/VDM_EVFI/train_bsergb" 




#--multi_gpu --num_processes 4
#module load anaconda3_gpu
#srun -A bfsm-delta-gpu \--partition=gpuA100x4 \--nodes=1 \--ntasks=1 \--gpus-per-node=2 \--mem=100G \--time=20:00:00  \--pty bash

#time left -- squeue -u $USER -o "%.10i %.12j %.2t %.10M %.10l %.10L"

#Screen alternative

#tmux new -s mysession
# Ctrl + b, then d
# tmux ls
# tmux kill-session -t mysession



#LEFT_RGB, RIGHT EVENT INPUT SHAPES
#  first left rgb shape is torch.Size([1, 3, 375, 375]) and last is torch.Size([1, 3, 375, 375]) 
#  eve_latenmst shape is torch.Size([1, 17, 6, 375, 375])

 

#VDM_EVFI INPUT SHAPES
# evenst shape is torch.Size([1, 14, 6, 320, 512])
# evenst shape is torch.Size([1, 14, 6, 320, 512])
# Steps:   0%|                                                         | 0/22800 [01:00<?, ?it/s, lr=5e-5, step_loss=0.16] pixel values rgbshape is torch.Size([1, 14, 3, 320, 512])
#  pixel values rgbshape is torch.Size([1, 14, 3, 320, 512])
#  conditional_pixel_values values rgbshape is torch.Size([1, 1, 3, 320, 512])
# -------------
#  conditional_pixel_values values rgbshape is torch.Size([1, 1, 3, 320, 512])
# -------------
# evenst shape is torch.Size([1, 14, 6, 320, 512])
# evenst shape is torch.Size([1, 14, 6, 320, 512])




#streo_output_errro_causing 

# inp_noisy_latents is torch.Size([1, 5, 8, 46, 46])
#  right events shaope is torch.Size([1, 5, 6, 375, 375])
# till here
# torch.Size([1, 5, 6, 375, 375])
# ----------
# (torch.Size([1, 5, 8, 46, 46]), torch.Size([1, 1, 1024]), torch.Size([1, 5, 6, 375, 375])) are input shapes to inpu_noisy, clip_hidden and event shapes
# sample shape is torch.Size([5, 320, 46, 46]) and evenst condition shape is torch.Size([5, 320, 47, 47])
# inp_noisy_latents is torch.Size([1, 5, 8, 46, 46])
#  right events shaope is torch.Size([1, 5, 6, 375, 375])
# till here
# torch.Size([1, 5, 6, 375, 375])
# ----------
# (torch.Size([1, 5, 8, 46, 46]), torch.Size([1, 1, 1024]), torch.Size([1, 5, 6, 375, 375])) are input shapes to inpu_noisy, clip_hidden and event shapes
# sample shape is torch.Size([5, 320, 46, 46]) and evenst condition shape is torch.Size([5, 320, 47, 47])


