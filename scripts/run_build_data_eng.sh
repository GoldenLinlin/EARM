python build_earm_data_en.py \
    --merge_folder /home/work/EARM/data/rm_outputs_eng \
    --output_dir ./outputs/earm_train_data_eng.json \
    --model_name /root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct \
    --min_edit_distance_diff 1 \
    --pairs_per_item 10