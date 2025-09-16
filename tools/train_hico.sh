export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port 3992 --use_env main.py \
    --batch_size 2 \
    --output_dir ./checkpoints/hico_det/cn_adapter_atten \
    --epochs 200 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 64 \
    --enable_dec \
    --num_tokens 12 \
    --dataset_file hico  --set_cost_hoi_type 5\
    --enable_focal_loss