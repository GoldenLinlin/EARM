import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict

class EditAwareRewardModel(nn.Module):
    """
    编辑感知奖励模型 (Edit-Aware Reward Model, EARM)
    
    特点:
    1. 输出整体奖励分数 R(x, y)
    2.  输出 token 级别分数用于加权损失
    3.  融合编辑掩码信息
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        pivot_token_weight: float = 4.0, 
        edit_token_weight: float = 2.0,
        normal_token_weight: float = 1.0
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path, config=self.config, trust_remote_code=True
        )
        
        if use_lora:
            self._setup_lora(lora_r, lora_alpha, lora_dropout)
        
        hidden_size = self.config.hidden_size
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.token_score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )

        # 初始化 Trick: 让初始输出接近 0，防止 Loss 爆炸
        for module in [self.value_head, self.token_score_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, std=0.001)
                    if layer.bias is not None: torch.nn.init.zeros_(layer.bias)
        
        self.pivot_token_weight = pivot_token_weight
        self.edit_token_weight = edit_token_weight
        self.normal_token_weight = normal_token_weight
    
    def _setup_lora(self, r, alpha, dropout):
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            print(f"LoRA Enabled: r={r}, alpha={alpha}")
        except ImportError:
            print("PEFT not installed, using full finetuning")
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def get_edit_weights(self, edit_mask: torch. Tensor) -> torch.Tensor:
        """
        将编辑掩码转换为权重
        edit_mask: 0=未编辑, 1=编辑起始(pivot), 2=编辑延续
        """
        weights = torch.where(
            edit_mask == 1, torch.full_like(edit_mask, self.pivot_token_weight, dtype=torch.float),
            torch.where(
                edit_mask == 2, torch.full_like(edit_mask, self.edit_token_weight, dtype=torch.float),
                torch.full_like(edit_mask, self.normal_token_weight, dtype=torch.float)
            )
        )
        return weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: Optional[torch.Tensor] = None,
        edit_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch. Tensor]:
        """
        前向传播
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state # [B, L, H]
        
        # Global Reward (Right Padding)
        batch_size = input_ids.size(0)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_states[torch.arange(batch_size), seq_lengths]
        reward = self.value_head(last_hidden).squeeze(-1)
        
        # Token Scores
        token_scores = self.token_score_head(hidden_states).squeeze(-1) # [B, L]
        
        weighted_score = None
        if edit_mask is not None:
            # 1. 基础权重
            base_weights = self.get_edit_weights(edit_mask)
            
            # 2. 构建有效计算区域 Mask (关键修正：屏蔽 Source 和 Padding)
            # 必须同时满足: 1. 不是Padding (attention_mask=1) 2. 不是Source (idx >= start_idx)
            valid_mask = attention_mask.clone()
            
            if response_start_idx is not None:
                seq_range = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
                is_response = seq_range >= response_start_idx.unsqueeze(1)
                valid_mask = valid_mask * is_response.long()
            
            # 3. 过滤权重 (Source和Padding的权重归零)
            final_weights = base_weights * valid_mask.float()
            
            # 4. 加权平均
            masked_scores = token_scores * final_weights
            weight_sum = final_weights.sum(dim=1) + 1e-8
            
            weighted_score = masked_scores.sum(dim=1) / weight_sum
        
        return {
            'reward': reward,
            'token_scores': token_scores,
            'weighted_score': weighted_score if weighted_score is not None else reward
        }

class EARMLoss(nn.Module):
    """
    EARM 混合损失函数
    
    L_EARM = L_Rank + λ * L_WeightedRank + μ * L_Margin
    """
    
    def __init__(
        self,
        lambda_weighted: float = 0.5,
        mu_margin: float = 0.1,
        margin: float = 1.0,
    ):
        super().__init__()
        self.lambda_weighted = lambda_weighted
        self.mu_margin = mu_margin
        self.margin = margin
    
    def forward(self, chosen_reward, rejected_reward, chosen_weighted_score=None, rejected_weighted_score=None):
        # 1. Rank Loss
        rank_loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
        
        # 2. Weighted Rank Loss
        weighted_rank_loss = torch.tensor(0.0, device=chosen_reward.device)
        if self.lambda_weighted > 0 and chosen_weighted_score is not None:
            weighted_rank_loss = -F.logsigmoid(chosen_weighted_score - rejected_weighted_score).mean()
            
        # 3. Margin Loss (原封不动加回来了)
        margin_loss = torch.tensor(0.0, device=chosen_reward.device)
        if self.mu_margin > 0:
            margin_loss = F.relu(self.margin - (chosen_reward - rejected_reward)).mean()
            
        # 总损失
        total_loss = rank_loss + self.lambda_weighted * weighted_rank_loss + self.mu_margin * margin_loss
        
        accuracy = (chosen_reward > rejected_reward).float().mean()
        
        return {
            'loss': total_loss,
            'rank_loss': rank_loss,
            'weighted_rank_loss': weighted_rank_loss,
            'margin_loss': margin_loss,
            'accuracy': accuracy.detach(),
            "chosen_reward_mean": chosen_reward.mean().detach(),
            "rejected_reward_mean": rejected_reward.mean().detach(),
            "reward_margin": (chosen_reward - rejected_reward).mean().detach(),
        }