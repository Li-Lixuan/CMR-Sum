lang=java
ckp_dir=java

#Training
CUDA_VISIBLE_DEVICES=0  python -u run.py \
	--do_train \
	--do_eval \
	--seed 625 \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/$lang/train.json \
	--dev_filename dataset/$lang/dev.json \
	--output_dir saved_models/$ckp_dir \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 13 \
	--eval_batch_size 13 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 70 
	
# Evaluating	
CUDA_VISIBLE_DEVICES=0 python -u run.py \
	--seed 625 \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename dataset/$lang/test.json \
	--train_filename dataset/$lang/train.json \
	--re_base_filename dataset/$lang/re_base.json \
	--output_dir saved_models/$ckp_dir \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 6 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 10 \
	--rate_re_base 1