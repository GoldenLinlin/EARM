import os
import json
import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm


def work(text):
    text = text.strip()
    text = text.replace('.', ' .')
    text = text.replace(',', ' ,')
    text = text.replace(':', ' :')
    text = text.replace('?', ' ?')
    text = text.replace('!', ' !')
    text = text.replace(';', ' ;')
    text = text.replace('"', ' " ')
    text = text.replace('\'', ' \' ')
    text = text.replace('n \' t', ' n\'t')
    text = text.replace('\' s', ' \'s')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    return text


# ===============================================================
# 1. 模型定义
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
        batch_size = input_ids.size(0)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = outputs.last_hidden_state[torch.arange(batch_size, device=input_ids.device), seq_lengths]
        return self.value_head(last_hidden).squeeze(-1)


# ===============================================================
# 2. 推理类
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
        
        head_path = os.path.join(ckpt_path, "value_head.bin")
        if os.path.exists(head_path):
            self.model.value_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        else:
            print("⚠️ 警告: 没找到 value_head.bin")
        
        self.model.eval().to(self.device)
        if torch.cuda.is_bf16_supported():
            self.model.bfloat16()
        else:
            self.model.half()

    @torch.no_grad()
    def score(self, source_text, responses, batch_size=8):
        all_scores = []
        input_ids_list = []
        attention_mask_list = []
        
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": source_text}], 
            tokenize=True, 
            add_generation_prompt=True
        )
        
        for resp in responses:
            resp_ids = self.tokenizer.encode(resp, add_special_tokens=False)
            full_ids = prompt_ids + resp_ids + [self.tokenizer.eos_token_id]
            input_ids_list.append(full_ids)
            attention_mask_list.append([1] * len(full_ids))
        
        for i in range(0, len(input_ids_list), batch_size):
            batch_ids = input_ids_list[i:i + batch_size]
            batch_masks = attention_mask_list[i:i + batch_size]
            max_len = max(len(x) for x in batch_ids)
            if max_len > 2048:
                max_len = 2048
            
            padded_ids = []
            padded_masks = []
            for ids, masks in zip(batch_ids, batch_masks):
                curr_len = len(ids)
                if curr_len > max_len:
                    padded_ids.append(ids[:max_len])
                    padded_masks.append(masks[:max_len])
                else:
                    pad_len = max_len - curr_len
                    padded_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                    padded_masks.append(masks + [0] * pad_len)
            
            input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor(padded_masks, dtype=torch.long).to(self.device)
            rewards = self.model(input_tensor, mask_tensor)
            
            if rewards.ndim == 0:
                all_scores.append(rewards.float().item())
            else:
                all_scores.extend(rewards.view(-1).float().cpu().numpy().tolist())
        
        return all_scores


# ===============================================================
# 3. 辅助函数
# ===============================================================
pmt = """Please identify if the provided sentence has any grammatical errors. If there are any, please correct them by making only the minimum necessary changes.
The modification requirements are very strict. Do not make changes related to fluency, politeness, sentence structure, colloquialisms, sentence length, awkwardness, or style, as these are considered stylistic optimizations and not objective grammatical errors.
If the sentence is grammatically correct, return the original sentence. Ensure that all modifications you make are based on established grammatical rules, not stylistic polishing.
Please provide your final answer in the following format.
<answer>
Your corrected sentence, or the original sentence if no errors are found.
</answer>
The sentence to modify is as follows:
"""


def process_generation(text, original_sentence):
    answer_trigger = "<answer>"
    try:
        if answer_trigger in text:
            gen_text = text.split(answer_trigger)[-1].split("</answer>")[0].strip()
            gen_text = gen_text.split("\n")[0].strip()
            gen_text = work(gen_text)
            if "<think>" in gen_text:
                return original_sentence.strip()
            return gen_text
        return original_sentence.strip()
    except:
        return original_sentence.strip()


# ===============================================================
# 4. 主程序 - TXT 输入输出版本 (支持 roll_values 复用)
# ===============================================================
if __name__ == "__main__":
    # 配置参数
    llm_model_path = "/home/work/eng_stage2_think"
    rm_model_path = "/home/work/EARM/earm_output_ENG_3loss_weight=1_1_1,MARGIN0.1,LAMBDA_WEIGHTED0.3_sft/best_model"
    input_file = "/home/work/EARM/bea.txt"       # 输入 txt，每行一个原句
    output_prefix = "/home/work/EARM/bea_output_-1_1_1"  # 输出前缀，会生成 bea_output_roll1.txt 等
    
    temp = 1.0
    batch_size = 5000
    rm_batch_size = 512
    roll_values = [16]  # 每个 roll 值都会输出一个文件
    cache_file = f"./generations_cache_t{temp}.json"  # 缓存文件
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
    
    # 读取输入文件 (每行一个句子)
    with open(input_file, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
    
    # data = data[:5]  # debug
    
    print(f"====== 读取到 {len(data)} 条数据 ======")
    
    # 初始化
    all_generations = {i: [] for i in range(len(data))}
    cumulative_roll = 0
    llm = None
    
    # 尝试加载缓存
    if os.path.exists(cache_file):
        print(f"====== 发现缓存文件: {cache_file}，正在加载... ======")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # 转换 key 为 int
            all_generations = {int(k): v for k, v in cached_data.items()}
            # 计算当前已生成数量
            if len(all_generations) > 0:
                cumulative_roll = len(all_generations[0])
            print(f"====== 成功加载缓存，当前已生成 {cumulative_roll} 条/样本 ======")
        except Exception as e:
            print(f"⚠️ 加载缓存失败 ({e})，将重新开始生成。")
            all_generations = {i: [] for i in range(len(data))}
            cumulative_roll = 0
    
    # 启动 RM
    rm = EarmRewardModel(rm_model_path, device="cuda:7")
    
    # 遍历每个 roll 值
    for target_roll in sorted(roll_values):
        need_generate = target_roll - cumulative_roll
        
        # --- 生成阶段 ---
        if need_generate > 0:
            # 按需启动 LLM
            if llm is None:
                print("====== 启动 LLM ======")
                llm = LLM(
                    model=llm_model_path, 
                    tensor_parallel_size=4, 
                    max_model_len=8192, 
                    gpu_memory_utilization=0.9, 
                    trust_remote_code=True
                )
            
            print(f"需要补充生成 {need_generate} 条 (目标: {target_roll}, 当前: {cumulative_roll})")
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_indices = list(range(i, min(i + batch_size, len(data))))
                
                # 构造输入
                batch_inputs = []
                for sentence in batch:
                    full_content = pmt + sentence
                    pt = [{'role': 'user', 'content': full_content}]
                    batch_inputs.append(tokenizer.apply_chat_template(pt, tokenize=False, add_generation_prompt=True))
                
                # 生成补充的候选
                for _ in range(need_generate):
                    outputs = llm.generate(
                        batch_inputs, 
                        SamplingParams(temperature=temp, max_tokens=5120, top_p=0.95, stop=["<|im_end|>"])
                    )
                    for idx, output in enumerate(outputs):
                        original_sentence = batch[idx]
                        processed = process_generation(output.outputs[0].text, original_sentence)
                        all_generations[batch_indices[idx]].append(processed)
            
            cumulative_roll = target_roll
            
            # 保存缓存
            print(f"====== 保存缓存到: {cache_file} ======")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(all_generations, f, ensure_ascii=False, indent=2)
        else:
            print(f"当前生成数量 ({cumulative_roll}) 已满足目标 ({target_roll})，跳过生成阶段。")
        
        # --- 评估阶段 (RM 选择最佳) ---
        print(f"====== 使用 RM 选择最佳候选 (Roll={target_roll}) ======")
        
        results = []
        for idx, sentence in enumerate(tqdm(data)):
            # 只取前 target_roll 个候选
            candidates = list(set(all_generations.get(idx, [])[:target_roll]))
            
            if not candidates:
                best_text = sentence
            else:
                scores = rm.score(sentence, candidates, batch_size=rm_batch_size)
                best_idx = int(np.argmax(scores))
                best_text = candidates[best_idx]
            
            results.append(best_text)
        
        # --- 输出到 txt ---
        output_file = f"{output_prefix}_roll{target_roll}_t{temp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in results:
                f.write(line + "\n")
        
        print(f"====== Roll={target_roll} 完成！结果已保存到: {output_file} ======")
    
    print("====== 全部完成！======")
