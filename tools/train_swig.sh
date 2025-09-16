export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port 3993 --use_env main.py \
  --batch_size 2 \
  --output_dir ./checkpoints/swig_hoi/cn_clip_alpha_atten \
  --epochs 120 \
  --lr 1e-4 --min-lr 1e-7 \
  --hoi_token_length 64 \
  --enable_dec \
  --num_tokens 12 \
  --dataset_file swig --set_cost_hoi_type 5 \
  --enable_focal_loss \
  #--resume checkpoints/swig_hoi/cn_clip_alpha_atten/checkpoint0095.pth