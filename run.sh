#python3 src/run_language_modeling.py --train_data_file resources/ai_domainlevel.txt \
#                                     --line_by_line \
#                                     --output_dir distilbert-test \
#                                     --model_type distilbert \
#                                     --tokenizer_name distilbert-base-cased \
#                                     --per_gpu_train_batch_size 16 \
#                                     --gradient_accumulation_steps 128  \
#                                     --model_name_or_path distilbert-base-cased \
#                                     --do_train \
#                                     --max_steps 12500  \
#                                     --learning_rate 0.0005 \
#                                     --mlm

python3 src/run_language_modeling.py --output_dir=distilbert-test \
                                     --model_type=distilbert \
                                     --model_name_or_path=distilbert-base-cased \
                                     --do_train \
                                     --train_data_file=resources/ai_domainlevel.txt \
                                     --mlm