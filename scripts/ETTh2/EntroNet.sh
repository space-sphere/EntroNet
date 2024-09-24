export CUDA_VISIBLE_DEVICES=0

model_name=EntroNet

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
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
    --itr 1

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/ETT/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --d_mutual 256 \
    --itr 1

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/ETT/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --d_mutual 128 \
    --use_entropy 1 \
    --itr 1