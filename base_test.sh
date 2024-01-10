basedir=""

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=8   --master_port 32711 ./train.py \
--pretrain_dir $basedir \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/ret-msrvtt-lr2e-5-bs64-epoch5'   \
--learning_rate 2e-5  \
--train_video_sample_num 4 \
--test_video_sample_num 8  \
--save_best true \