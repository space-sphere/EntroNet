export CUDA_VISIBLE_DEVICES=0

model_name=EntroNet

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --freq 15min \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 512 \
    --d_mutual 256 \
    --use_entropy 0 \
    --itr 1