# export CUDA_VISIBLE_DEVICES=0,1,2,3
# nohup python -m torch.distributed.launch --nproc_per_node=4 \
# --master_port=12138 --use_env run_train.py --diff_steps 2000 \
# --microbatch 100 --lr 0.0001 --learning_steps 55000 --save_interval 1000 \
# --seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
# --dataset qqp --data_dir datasets/QQP --vocab bert --seq_len 128 \
# --simi_penalty cosine --simi_lambda 2 --simi_step 100 \
# --resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test_ori20221113-20:27:29/ema_0.9999_050000.pt \
# --schedule_sampler lossaware --notes qqp > train_logs/qqp.log 2>&1 &


# OPENAI_LOGDIR=diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qqp20221126-17:27:40  \
# TOKENIZERS_PARALLELISM=false python train.py   --checkpoint_path diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qqp20221126-17:27:40 \
# --dataset qqp --data_dir datasets/QQP --vocab bert --use_plm_init no --lr 0.0001 --batch_size 200 --microbatch 200 --diffusion_steps 2000 --noise_schedule sqrt \
# --schedule_sampler lossaware --resume_checkpoint none --seq_len 128 --hidden_t_dim 128 --seed 102 --hidden_dim 128 --learning_steps 50 --save_interval 10000 \
# --config_name bert-base-uncased --notes qqp20221126-17:27:40 --simi_lambda 0.01 --simi_step 10 --simi_penalty cosine 

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=11411 --use_env run_train.py --diff_steps 2000 \
--microbatch 70 --lr 0.0001 --learning_steps 55000 --save_interval 1000 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 70 \
--dataset qqp --data_dir datasets/QQP --vocab bert --seq_len 128 \
--simi_penalty cosine --near_step 50 --far_step 100 \
--near_lambda 0.5 --far_lambda 2 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test_ori20221113-20:27:29/ema_0.9999_050000.pt \
--schedule_sampler lossaware --notes qqp 

wait

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12138 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 100000 --save_interval 1000 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset qqp --data_dir datasets/QQP --vocab bert --seq_len 128 \
--simi_penalty cosine --simi_lambda 2 --simi_step 100 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test_ori20221113-20:27:29/ema_0.9999_050000.pt \
--schedule_sampler lossaware --notes qqp > train_logs/near_large.log &

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12138 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 320000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset qqp --data_dir datasets/QQP --vocab bert --seq_len 128 \
--simi_penalty kl --simi_lambda 100 --simi_step 100 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_240000.pt \
--schedule_sampler lossaware --notes qqp > train_logs/kl.log &


export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12138 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 320000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset roc --data_dir datasets/ROCstory --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -2 --simi_step 10 --simi_noise 0.05 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_240000.pt \
--schedule_sampler lossaware --notes roc > train_logs/roc_final_more.log &

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12238 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 320000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset roc --data_dir datasets/ROCstory --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -1 --simi_step 10 --simi_noise 0.05 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_240000.pt \
--schedule_sampler lossaware --notes roc > train_logs/roc_final_alpha1.log &


export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12338 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 180000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset qg --data_dir datasets/QG --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -1 --simi_step 10 --simi_noise 0.05 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_qg_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qg20221210-20:52:13/ema_0.9999_060000.pt \
--schedule_sampler lossaware --notes qg > train_logs/qg_final_alpha1.log &

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=13338 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 180000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset qg --data_dir datasets/QG --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -2 --simi_step 10 --simi_noise 0.1 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_qg_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qg20221210-20:52:13/ema_0.9999_060000.pt \
--schedule_sampler lossaware --notes qg > train_logs/qg_final_noise.1.log &

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=14338 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 180000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset qg --data_dir datasets/QG --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -2 --simi_step 10 --simi_noise 0.2 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_qg_h128_lr0.0001_t2000_sqrt_lossaware_seed102_qg20221210-20:52:13/ema_0.9999_060000.pt \
--schedule_sampler lossaware --notes qg > train_logs/qg_final_noise.2.log &

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12238 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 320000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset roc --data_dir datasets/ROCstory --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -1 --simi_step 10 --simi_noise 0.005 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_240000.pt \
--schedule_sampler lossaware --notes roc > train_logs/roc_final_noise0.005_alpha1.log &

export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=14238 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 320000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset roc --data_dir datasets/ROCstory --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -1 --simi_step 10 --simi_noise 0.105 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_240000.pt \
--schedule_sampler lossaware --notes roc > train_logs/roc_final_noise0.105_alpha1.log &

export CUDA_VISIBLE_DEVICES=0,1,2,3,
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=11238 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 70000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset roc --data_dir datasets/ROCstory --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -1 --simi_step 10 --simi_noise 0.05 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_040000.pt \
--schedule_sampler lossaware --notes roc > train_logs/roc_final_alpha_from4w.log &

export CUDA_VISIBLE_DEVICES=0,1,2,3,
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=13238 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 110000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset roc --data_dir datasets/ROCstory --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -1 --simi_step 10 --simi_noise 0.05 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_080000.pt \
--schedule_sampler lossaware --notes roc > train_logs/roc_final_alpha_from8w.log &

export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=12238 --use_env run_train.py --diff_steps 2000 \
--microbatch 100 --lr 0.0001 --learning_steps 150000 --save_interval 2500 \
--seed 109 --noise_schedule sqrt --hidden_dim 128 --bsz 100 \
--dataset roc --data_dir datasets/ROCstory --vocab bert --seq_len 128 \
--simi_penalty l2_noise_random --simi_lambda -1 --simi_step 10 --simi_noise 0.05 \
--resume_checkpoint /opt/data/private/DiffuSeq-posttrain/diffusion_models/diffuseq_roc_h128_lr0.0001_t2000_sqrt_lossaware_seed102_roc20221211-01:47:59/ema_0.9999_120000.pt \
--schedule_sampler lossaware --notes roc > train_logs/roc_final_alpha_from12w.log &