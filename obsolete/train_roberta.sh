export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=6,7
export DATA_DIR=./squad_v2
export MODEL_DIR=./roberta-large/pretrained_model
export OUTPUT_DIR=./albert-xxlarge-v2/finetuned_ckpt

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
run_my_squad.py \
    --model_type albert \
    --model_name_or_path $MODEL_DIR \
    --data_dir $DATA_DIR \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $DATA_DIR/train-v2.0.json \
    --predict_file $DATA_DIR/dev-v2.0.json \
    --per_gpu_train_batch_size 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --save_steps 5000 \
    --threads 4 \
    --version_2_with_negative \
    --overwrite_output_dir
