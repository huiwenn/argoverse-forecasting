python const_vel_train_test.py --test_features features/train.pkl --obs_len 20 --pred_len 30 --traj_save_path traj/const_vel.pkl


  python lstm_train_test.py  --test_features features/val.pkl --train_features features/train.pkl  --val_features features/val.pkl --use_delta --normalize --test --obs_len 20 --pred_len 30 --model_path saved_models/lstm/LSTM_rollout30.pth.tar --traj_save_path traj/lstm_none.pkl

python eval_forecasting_helper.py --metrics --gt features/val_gt.pkl --forecast  traj/lstm_none.pkl --horizon 30 --obs_len 20 --miss_threshold 2 --features features/val.pkl --max_n_guesses 1 

python eval_forecasting_helper.py --viz --metrics --gt features/val_gt.pkl --forecast  traj/lstm_none.pkl --horizon 30 --obs_len 20 --miss_threshold 2 --features features/val.pkl --max_n_guesses 1 --viz_seq_id features/indexviz.pkl

## no use delta

 python prob_lstm_train_test.py --metric nll --test_features features/val.pkl --train_features features/train.pkl  --val_features features/val.pkl   --normalize --obs_len 20 --pred_len 30 --save_path model/nll_no_delta --traj_save_path traj/nll_no_delta.pkl

 python prob_lstm_train_test.py --metric nll --test --test_features features/val.pkl --train_features features/train.pkl  --val_features features/val.pkl  --use_delta --normalize --obs_len 20 --pred_len 30 --model_path model/nll_new/LSTM_rollout30.pth.tar --traj_save_path traj/lstm_nll_new.pkl

python prob_lstm_train_test.py --metric mis --test_features features/val.pkl --train_features features/train.pkl  --val_features features/val.pkl  --normalize --obs_len 20 --pred_len 30 --save_path model/mis_no_delta --traj_save_path traj/mis_no_delta.pkl 

 python prob_lstm_train_test.py --metric mis --test_features features/val.pkl --train_features features/train.pkl  --val_features features/val.pkl  --use_delta --normalize --obs_len 20 --pred_len 30 --save_path model/nis_500 --traj_save_path traj/lstm_mis_500.pkl --model_path model/nis_500/LSTM_rollout10.pth.tar

python prob_lstm_train_test.py --metric mis --test_features features/val.pkl --train_features features/train.pkl  --val_features features/val.pkl  --use_delta --normalize --obs_len 20 --pred_len 30 --save_path model/mis_scale100 --traj_save_path traj/lstm_mis_scale100.pkl

## PECCO
python scripts/train.py --dataset_path ../argoverse_data --rho-reg --model_name rho_reg_pecco --batch_size 4 --train
