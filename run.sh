python3 src/run_mlm_xlmr.py --train_file resources/esco_features.json \
                            --line_by_line \
                            --output_dir xlm-test \
                            --model_type xlm-roberta-large \
                            --tokenizer_name xlm-roberta-large \
                            --per_device_train_batch_size 4 \
                            --per_device_eval_batch_size 4 \
                            --gradient_accumulation_steps 16  \
                            --eval_accumulation_steps 16 \
                            --gradient_checkpointing \
                            --model_name_or_path xlm-roberta-large \
                            --tf32 1 \
                            --do_train \
                            --do_eval \
                            --evaluation_strategy steps \
                            --max_steps 200000  \
                            --save_steps 2000 \
                            --learning_rate 0.0005 \
                            --logging_steps 100 \
                            --eval_steps 500 \
                            --save_total_limit 5
#                            \
#                            --prediction_loss_only
