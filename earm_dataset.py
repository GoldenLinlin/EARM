import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List
import json

class EARMDataset(Dataset):
    """EARM 训练数据集 (ID拼接版)"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        # 强制右侧 Padding，这对 Reward Model 至关重要
        self.tokenizer.padding_side = "right" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length
        
        print(f"加载数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"数据量: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        
        # 直接使用预计算的 IDs
        chosen_input = self._build_input_from_ids(
            item['source'], item['chosen_ids'], item['chosen_edit_mask']
        )
        rejected_input = self._build_input_from_ids(
            item['source'], item['rejected_ids'], item['rejected_edit_mask']
        )
        
        return {
            'chosen_input': chosen_input,
            'rejected_input': rejected_input,
        }
    
    def _build_input_from_ids(self, source: str, response_ids: List[int], edit_mask: List[int]) -> Dict:
        """
        拼接: [Prompt IDs] + [Response IDs (from json)] + [EOS]
        """
        # 1. 生成 Prompt IDs (使用 Chat Template)
        messages = [{"role": "user", "content": source}]
        # tokenize=True 返回 ID 列表, add_generation_prompt=True 加上 <|im_start|>assistant
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        
        # 2. 计算 Response 起始位置
        response_start_idx = len(prompt_ids)
        
        # 3. 拼接 IDs
        # response_ids 是我们在生成数据时算好的，绝对纯净
        full_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        
        # 4. 拼接 Mask
        # Prompt和EOS部分的 Mask 为 0
        full_edit_mask = [0] * len(prompt_ids) + edit_mask + [0]
        
        # 5. Padding 和 Truncation
        curr_len = len(full_ids)
        
        if curr_len > self.max_length:
            # 截断 (保留头部)
            input_ids = torch.tensor(full_ids[:self.max_length], dtype=torch.long)
            final_edit_mask = torch.tensor(full_edit_mask[:self.max_length], dtype=torch.long)
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            # Padding (补在右侧)
            pad_len = self.max_length - curr_len
            
            # 转 Tensor
            input_tensor = torch.tensor(full_ids, dtype=torch.long)
            mask_tensor = torch.tensor(full_edit_mask, dtype=torch.long)
            
            # 拼接 Padding
            input_ids = torch.cat([
                input_tensor, 
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            final_edit_mask = torch.cat([
                mask_tensor,
                torch.zeros((pad_len,), dtype=torch.long)
            ])
            attention_mask = torch.cat([
                torch.ones(curr_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ])
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'response_start_idx': response_start_idx,
            'edit_mask': final_edit_mask
        }

class EARMDataCollator:
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            'chosen_input_ids': torch.stack([x['chosen_input']['input_ids'] for x in batch]),
            'chosen_attention_mask': torch.stack([x['chosen_input']['attention_mask'] for x in batch]),
            'chosen_response_start_idx': torch.tensor([x['chosen_input']['response_start_idx'] for x in batch]),
            'chosen_edit_mask': torch.stack([x['chosen_input']['edit_mask'] for x in batch]),
            
            'rejected_input_ids': torch.stack([x['rejected_input']['input_ids'] for x in batch]),
            'rejected_attention_mask': torch.stack([x['rejected_input']['attention_mask'] for x in batch]),
            'rejected_response_start_idx': torch.tensor([x['rejected_input']['response_start_idx'] for x in batch]),
            'rejected_edit_mask': torch.stack([x['rejected_input']['edit_mask'] for x in batch]),
        }