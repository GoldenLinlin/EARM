python build_earm_data.py \
    --merge_folder ./data/rm_valid_with_list \
    --output_dir ./outputs/earm_val_data.json \
    --model_name /root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct \
    --min_edit_distance_diff 1 \
    --pairs_per_item 3