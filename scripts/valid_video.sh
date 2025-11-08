accelerate launch valid.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="/u/nthadishetty/VDM_EVFI/scripts" \
    --test_data_path="/u/nthadishetty/VDM_EVFI_changed/example_data/clear_motion_sample" \
    --event_scale=1 \
    --num_frames=13 \
    --video_mode \
    --rescale_factor=1 \
    --controlnet_model_name_or_path="/u/nthadishetty/VDM_EVFI_changed/13_frames" \
    --eval_folder_start=0 \
    --eval_folder_end=-1 \
    --width=240 \
    --height=240 \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=4 \
    --num_train_epochs=50 \
    --checkpointing_steps=20 \
    --checkpoints_total_limit=1 \
    --learning_rate=3e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --validation_steps=1 \
    --num_workers=8 \
    --num_inference_steps=2 \
    --decode_chunk_size=1 \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision="fp16" \
    --overlapping_ratio=0.1 \
    --t0=0 \
    --M=2 \
    --s_churn=0.5 \
    
#module load anaconda3_gpu
#srun -A bfsm-delta-gpu \--partition=gpuA100x4 \--nodes=1 \--ntasks=1 \--gpus-per-node=2 \--mem=100G  \--pty bash

srun --account=bfsm-delta-gpu --partition=cpu-interactive \
  --nodes=1 --tasks=1 --tasks-per-node=1 \
  --cpus-per-task=4 --mem=16g \
  --pty bash