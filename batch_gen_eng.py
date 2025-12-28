import os
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Prompt for the model


if __name__ == "__main__":
    # --- Configuration ---
    model_base_path = '/home/work/LLaMA-Factory/output/eng_stage2_only_qwen7b'
    input_folder_path = "/home/work/EARM/data/data_gen_eng"
    output_folder_path = "./eng_stage2_only_qwen7b_valid/"
    repeat_times = 8
    temperature = 1
    batch_size = 540000  # 每批最大句数

    os.makedirs(output_folder_path, exist_ok=True)

    # --- Load Tokenizer and Model ---
    print(f"Loading tokenizer from: {model_base_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_base_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from: {model_base_path} using vLLM")
    llm = LLM(
        model=model_base_path,
        tensor_parallel_size=4,
        max_model_len=600,
        gpu_memory_utilization=0.9
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=600,
        top_p=0.95
    )
    print("Model and tokenizer loaded successfully.\n")

    input_files = [f for f in os.listdir(input_folder_path) if f.endswith("2.json")]

    for input_filename in input_files:
        current_input_file_path = os.path.join(input_folder_path, input_filename)
        eval_file_name_base = os.path.splitext(input_filename)[0]
        output_filename = f"{eval_file_name_base}.json"
        current_output_file_path = os.path.join(output_folder_path, output_filename)

        print(f"Processing file: {current_input_file_path}")

        # --- 读取所有句子 ---
        sentences = []
        ans = []
        try:
            with open(current_input_file_path, "r", encoding="utf-8") as f:
                f=f.read()
                f=json.loads(f)
                for line in f:
                    t = line['ans']
                    s = line['ori']
                    pmt = "Rewrite the input text into grammatically correct text. ### input:\n"
                    s=pmt+s+"\n\n"
                    ans.append(t)
                    sentences.append(s)
        except Exception as e:
            print(f"Error reading file {current_input_file_path}: {e}")
            continue

        if not sentences:
            print(f"No valid data to process in {input_filename}.")
            continue

        # --- 构建所有句子的完整 prompt ---
        prompts = []
        for s in sentences:
            prompt_text = s
            formatted_prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt_text}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(formatted_prompt)
            # conda activate verl
            # cd reward/
# CUDA_VISIBLE_DEVICES=4
        # --- 初始化结果列表 ---
        # 每个元素是 {"ori": 原句, "tgt": [16个结果]}
        all_results = [{"ori": s.split("### input:\n")[-1].strip(), "tgt": [],"ans": t} for s,t in zip( sentences,ans)]

        # --- 重复 16 次全量 batch 推理 ---
        for r in range(repeat_times):
            print(f"\n=== Round {r + 1}/{repeat_times}: Generating for {len(prompts)} sentences ===")

            # 分批避免 OOM
            round_outputs = []
            for start in range(0, len(prompts), batch_size):
                sub_batch = prompts[start:start + batch_size]
                try:
                    model_outputs = llm.generate(sub_batch, sampling_params)
                    for output in model_outputs:
                        round_outputs.append(output.outputs[0].text.strip())
                except Exception as e:
                    print(f"Error during generation in batch: {e}")
                    round_outputs.extend([f"MODEL_ERROR: {e}"] * len(sub_batch))

            # 汇总这一轮结果到 all_results 中
            for i, text in enumerate(round_outputs):
                all_results[i]["tgt"].append(text)

        # --- 保存 JSON ---
        try:
            with open(current_output_file_path, "w", encoding="utf-8") as f_out:
                json.dump(all_results, f_out, ensure_ascii=False, indent=2)
            print(f"\nSuccessfully wrote JSON to: {current_output_file_path}")
        except Exception as e:
            print(f"Error writing to output file {current_output_file_path}: {e}")

    print("\nAll files processed.")
