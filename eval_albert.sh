export CUDA_VISIBLE_DEVICES=0,2,3
export DATA_DIR=./squad_v2
export MODEL_DIR=./albert-xxlarge-v2/finetuned_ckpts/checkpoint-125000
export OUTPUT_DIR=$MODEL_DIR

python run_squad.py \
    --model_type albert \
    --model_name_or_path $MODEL_DIR \
    --data_dir $DATA_DIR \
    --do_eval \
    --do_lower_case \
    --predict_file dev-v2.0.json \
    --per_gpu_eval_batch_size=24 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --threads 4 \
    --version_2_with_negative \
    --overwrite_output_dir
