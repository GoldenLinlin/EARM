import os
import torch
import json
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

# ===============================================================
# 1. 模型定义 (保持不变)
# ===============================================================
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

# ===============================================================
# 2. 推理类 (严格对齐 EARMDataset)
# ===============================================================
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
pmt = "纠正输入句子中的语法错误，并输出正确的句子。### input:\n"

def process_generation(text, original_sentence):
    # answer_trigger = "### response:\n"
    print(text)
    # assert(0)
    
    return text.strip().replace(" ", "")

# ===============================================================
# 4. 主程序
# ===============================================================
if __name__ == "__main__":
    llm_model_path = "/home/work/models/nothink_64_32_600_rm"
    rm_model_path = "/home/work/EARM/best_model"  
    input_file = "./test.json"
    temp = 1
    batch_size = 3000
    rm_batch_size = 4096 
    
    # [新增] 定义缓存文件路径
    cache_file = f"./earm_generations_nothink_cache_t{temp}.json"

    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    
    # 1. 启动生成模型 (LLM)
    llm = LLM(model=llm_model_path, tensor_parallel_size=4, max_model_len=8192, gpu_memory_utilization=0.9, trust_remote_code=True)
    
    # 2. 启动奖励模型 (RM)
    rm = EarmRewardModel(rm_model_path, device="cuda:7")

    with open(input_file, "r") as f: data = json.load(f)
    # data = data[:5] # debug

    roll_values = [16]
    all_generations = {}
    cumulative_roll = 0

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
            print(f"⚠️ 加载缓存失败 ({e})，将重新开始生成。")
            all_generations = {}
            cumulative_roll = 0
    else:
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
                    full_content = pmt + item["Sentence"] + '\n\n'
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

        outfile = f"./earm_best_of_{target_roll}_{os.path.basename(llm_model_path)}_nothink_64_32_600_rm_t{temp}.json"
        with open(outfile, "w") as f: json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved: {outfile}")