export CUDA_VISIBLE_DEVICES=0

model_name=EntroNet

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/exchange_rate/ \
    --data_path exchange_rate.csv \
    --data Exchange \
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
    --d_ff 256 \
    --d_mutual 256 \
    --lradj 'TST' \
    --criterion 'mse' \
    --batch_size 128 \
    --train_epochs 30 \
    --learning_rate 0.0005 \
    --patience 5 \
    --itr 1

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/exchange_rate/ \
    --data_path exchange_rate.csv \
    --data Exchange \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 256 \
    --d_mutual 256 \
    --lradj 'TST' \
    --criterion 'mse' \
    --batch_size 128 \
    --train_epochs 30 \
    --learning_rate 0.0005 \
    --patience 5 \
    --itr 1

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/exchange_rate/ \
    --data_path exchange_rate.csv \
    --data Exchange \
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
    --lradj 'TST' \
    --criterion 'mae' \
    --batch_size 128 \
    --train_epochs 20 \
    --learning_rate 0.0005 \
    --patience 5 \
    --itr 1

# python3 -u run_longExp.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./data/ \
#     --data_path exchange_rate/exchange_rate.csv \
#     --data Exchange \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --dropout 0.2 \
#     --d_model 128 \
#     --d_ff 256 \
#     --d_mutual 256 \
#     --use_entropy 1 \
#     --itr 1

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/ \
    --data_path exchange_rate/exchange_rate.csv \
    --data Exchange \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --dropout 0.2 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 512 \
    --d_mutual 256 \
    --use_entropy 1 \
    --mutual_type 'gin' \
    --itr 1

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/ \
    --data_path exchange_rate/exchange_rate.csv \
    --data Exchange \
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
    --dropout 0.2 \
    --d_ff 256 \
    --d_mutual 256 \
    --use_entropy 0 \
    --itr 1

python3 -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/ \
    --data_path exchange_rate/exchange_rate.csv \
    --data Exchange \
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
    --dropout 0.2 \
    --d_ff 256 \
    --d_mutual 256 \
    --use_entropy 1 \
    --itr 1