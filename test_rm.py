import os
import torch
import json
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

class EditAwareRewardModel(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config, trust_remote_code=True)
        hidden_size = self.config.hidden_size
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        # Right Padding 取最后一个 token
        batch_size = input_ids.size(0)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = outputs.last_hidden_state[torch.arange(batch_size, device=input_ids.device), seq_lengths]
        return self.value_head(last_hidden).squeeze(-1)

class EarmRewardModel:
    def __init__(self, ckpt_path, device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[PAD]'

        print(f"正在加载 EARM: {ckpt_path}")
        self.model = EditAwareRewardModel(ckpt_path)
        
        # 加载 Head
        head_path = os.path.join(ckpt_path, "value_head.bin")
        if os.path.exists(head_path):
            self.model.value_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        else:
            print("⚠️ 警告: 没找到 value_head.bin")

        self.model.eval().to(self.device)
        if torch.cuda.is_bf16_supported(): self.model.bfloat16()
        else: self.model.half()

    @torch.no_grad()
    def score(self, source_text, responses, batch_size=8):
        """
        source_text: 对应训练时的 item['source'] (即原句)
        responses: 对应训练时的 response_ids
        """
        all_scores = []
        input_ids_list = []
        attention_mask_list = []

        # ------------------------------------------------------------------
        # 步骤 1: 构造 Prompt 部分
        # ------------------------------------------------------------------
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": source_text}], 
            tokenize=True, 
            add_generation_prompt=True
        )

        for resp in responses:
            # --------------------------------------------------------------
            # 步骤 2: 构造 Response 部分
            # --------------------------------------------------------------
            resp_ids = self.tokenizer.encode(resp, add_special_tokens=False)
            
            # 拼接: [Prompt Chat Template] + [Response] + [EOS]
            full_ids = prompt_ids + resp_ids + [self.tokenizer.eos_token_id]
            
            input_ids_list.append(full_ids)
            # Attention Mask 全为 1
            attention_mask_list.append([1] * len(full_ids))

        # ------------------------------------------------------------------
        # 步骤 3: Batch Forward (需要 Right Padding)
        # ------------------------------------------------------------------
        for i in range(0, len(input_ids_list), batch_size):
            batch_ids = input_ids_list[i : i + batch_size]
            batch_masks = attention_mask_list[i : i + batch_size]
            
            max_len = max(len(x) for x in batch_ids)
            if max_len > 2048: max_len = 2048 # 保护显存

            padded_ids = []
            padded_masks = []
            
            for ids, masks in zip(batch_ids, batch_masks):
                curr_len = len(ids)
                if curr_len > max_len:
                    padded_ids.append(ids[:max_len])
                    padded_masks.append(masks[:max_len])
                else:
                    pad_len = max_len - curr_len
                    # Right Padding
                    padded_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                    padded_masks.append(masks + [0] * pad_len)

            input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor(padded_masks, dtype=torch.long).to(self.device)
            
            rewards = self.model(input_tensor, mask_tensor)
            
            if rewards.ndim == 0: all_scores.append(rewards.float().item())
            else: all_scores.extend(rewards.view(-1).float().cpu().numpy().tolist())
            
        return all_scores

# ===============================================================
# 3. 辅助函数
# ===============================================================
pmt = "请识别我提供的句子是否有语法错误，如果有语法错误，请进行改正，请做出最少的修改。修改要求很严格，不要将流畅性，礼貌性，结构性，口语化，长短句，拗口性，风格等不属于语法范畴，而属于可优化的问题进行修改。如果没有错误，请回复原句。请保证你所做的修改都是有语法依据的，不要润色句子。\n最终答案请你按照如下格式回复。\n<answer>\n你修改后的句子，或者原句\n</answer>\n你要修改的句子如下：\n"

def process_generation(text, original_sentence):
    answer_trigger = "<answer>"
    try:
        if answer_trigger in text:
            gen_text = text.split(answer_trigger)[-1].split("</answer>")[0].strip()
            # 清理可能的思维链和其他杂质
            gen_text = gen_text.split("\n")[0].strip()
            if "<think>" in gen_text: return original_sentence.strip().replace(" ", "")
            return gen_text.replace(" ", "")
        return original_sentence.strip().replace(" ", "")
    except:
        return original_sentence.strip().replace(" ", "")

# ===============================================================
# 4. 主程序
# ===============================================================
if __name__ == "__main__":
    llm_model_path = "<BASE_MODEL_PATH>"
    rm_model_path = "<EARM_CHECKPOINT_PATH>"  
    input_file = "./test.json"
    temp = 1.0
    batch_size = 3000
    rm_batch_size = 512 
    
    # [新增] 定义缓存文件路径
    cache_file = f"./earm_generations_cache_t{temp}.json"

    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    
    # 1. 启动生成模型 (LLM)
    
    # 2. 启动奖励模型 (RM)
    rm = EarmRewardModel(rm_model_path, device="cuda:7")

    with open(input_file, "r") as f: data = json.load(f)
    # data = data[:5] # debug

    roll_values = [16]
    all_generations = {}
    cumulative_roll = 0
    llm=None
    # [新增] 尝试加载缓存
    if os.path.exists(cache_file):
        print(f"====== 发现缓存文件: {cache_file}，正在加载... ======")
        try:
            with open(cache_file, "r") as f:
                all_generations = json.load(f)
            
            # 计算当前缓存的 rollout 数量 (取第一个数据的生成列表长度作为参考)
            if len(all_generations) > 0:
                first_key = next(iter(all_generations))
                cumulative_roll = len(all_generations[first_key])
                print(f"====== 成功加载缓存，当前已生成 {cumulative_roll} 条/样本 ======")
        except Exception as e:
            llm = LLM(model=llm_model_path, tensor_parallel_size=4, max_model_len=8192, gpu_memory_utilization=0.9, trust_remote_code=True)

            print(f"⚠️ 加载缓存失败 ({e})，将重新开始生成。")
            all_generations = {}
            cumulative_roll = 0
    else:
        llm = LLM(model=llm_model_path, tensor_parallel_size=4, max_model_len=8192, gpu_memory_utilization=0.9, trust_remote_code=True)

        print(f"====== 未发现缓存文件，将开始新的生成 ======")

    for target_roll in sorted(roll_values):
        need_generate = target_roll - cumulative_roll
        
        # --- 生成阶段 (LLM) ---
        if need_generate > 0:
            print(f"需要补充生成 {need_generate} 条 (目标: {target_roll}, 当前: {cumulative_roll})")
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                batch_inputs = []
                for item in batch:
                    # LLM 需要指令 Prompt
                    full_content = pmt + item["Sentence"] + '\n'
                    pt = [{'role': 'user', 'content': full_content}]
                    batch_inputs.append(tokenizer.apply_chat_template(pt, tokenize=False, add_generation_prompt=True))

                for _ in range(need_generate):
                    outputs = llm.generate(batch_inputs, SamplingParams(temperature=temp, max_tokens=5120, top_p=0.95, stop=["<|im_end|>"]))
                    for idx, output in enumerate(outputs):
                        uuid = batch[idx]['﻿UUID']
                        processed = process_generation(output.outputs[0].text, batch[idx]["Sentence"])
                        if uuid not in all_generations: all_generations[uuid] = []
                        all_generations[uuid].append(processed)
            
            cumulative_roll = target_roll
            
            # [新增] 保存缓存
            print(f"====== 正在保存生成结果到缓存: {cache_file} ======")
            with open(cache_file, "w") as f:
                json.dump(all_generations, f, ensure_ascii=False, indent=4)
        else:
            print(f"当前生成数量 ({cumulative_roll}) 已满足目标 ({target_roll})，跳过生成阶段。")

        # --- 评估阶段 (RM) ---
        print(f"Ranking with RM (Roll={target_roll})...")
        results = {}

        for item in tqdm(data):
            uuid = item['﻿UUID']
            sentence = item["Sentence"]
            candidates = list(set(all_generations.get(uuid, [])[:target_roll]))

            if not candidates:
                best_text = sentence
            else:
                # ==========================================================
                # 关键修正：RM 输入不带 Prompt，只带原句 (Source)
                # 这与 EARMDataset 中的 item['source'] 对应
                # ==========================================================
                rm_source_input = sentence 
                
                # RM 内部会做: ChatTemplate(User=sentence) + Candidate + EOS
                scores = rm.score(rm_source_input, candidates, batch_size=rm_batch_size)
                
                best_idx = int(np.argmax(scores))
                best_text = candidates[best_idx]

            results[uuid] = {"error_flag": 0, "error_type": "*", "correction": best_text}

        outfile = f"./earm_best_of_{target_roll}_{os.path.basename(llm_model_path)}_weight=6_3_1,MARGIN0.1,LAMBDA_WEIGHTED0.3_test_t{temp}.json"
        with open(outfile, "w") as f: json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved: {outfile}")