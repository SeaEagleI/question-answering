export CUDA_VISIBLE_DEVICES=2
export DATA_DIR=./squad_v2
export MODEL_DIR=./albert-xxlarge-v2/pretrained_model
export OUTPUT_DIR=./albert-xxlarge-v2/finetuned_ckpts

python run_squad.py \
    --model_type albert \
    --model_name_or_path $MODEL_DIR \
    --data_dir $DATA_DIR \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file train-v2.0.json \
    --predict_file dev-v2.0.json \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=16 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --save_steps 5000 \
    --threads 4 \
    --version_2_with_negative \
    --overwrite_output_dir \
    --fp16
