export NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=1,2,5
export DATA_DIR=./newsqa_v1
export OUTPUT_DIR=./albert-xxlarge-v2/finetuned_ckpts_newsqa
export MODEL_DIR=./albert-xxlarge-v2/pretrained_model

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
newsqa_scripts/run_newsqa.py \
    --model_type albert \
    --model_name_or_path $MODEL_DIR \
    --data_dir $DATA_DIR \
    --do_train \
    --do_lower_case \
    --train_file train-v1.0.json \
    --predict_file dev-v1.0.json \
    --per_gpu_train_batch_size=6 \
    --per_gpu_eval_batch_size=32 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --save_steps 3000 \
    --threads 4 \
    --version_2_with_negative \
    --overwrite_output_dir \
    --fp16
