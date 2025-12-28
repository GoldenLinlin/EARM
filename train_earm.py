import os
import json
import argparse
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import wandb

from earm_model import EditAwareRewardModel, EARMLoss
from earm_dataset import EARMDataset, EARMDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="训练 EARM")

    # 数据参数
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=600)

    # 模型参数
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # 编辑权重参数
    parser.add_argument("--pivot_token_weight", type=float, default=10.0)
    parser.add_argument("--edit_token_weight", type=float, default=5.0)
    parser.add_argument("--normal_token_weight", type=float, default=1.0)

    # 损失函数参数
    parser.add_argument("--lambda_weighted", type=float, default=0.5)
    parser.add_argument("--mu_margin", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 输出参数
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="EARM")

    return parser.parse_args()


def save_model(model, tokenizer, accelerator, output_dir):
    """保存模型的通用函数"""
    os.makedirs(output_dir, exist_ok=True)

    # 解包模型 (unwrap)，去除 DDP/Accelerator 的包装
    unwrapped_model = accelerator.unwrap_model(model)
    
    # 保存 Backbone (如果是 LoRA，这会保存 adapter_model.bin)
    try:
        unwrapped_model.backbone.save_pretrained(output_dir)
        if accelerator.is_main_process:
            print(f"Backbone/LoRA weights saved to {output_dir}")
    except Exception as e: 
        if accelerator.is_main_process: 
            print(f"Warning: Backbone save failed: {e}")

    # 保存 Tokenizer
    try: 
        tokenizer.save_pretrained(output_dir)
        if accelerator.is_main_process: 
            print(f"Tokenizer saved to {output_dir}")
    except Exception as e:
        if accelerator.is_main_process: 
            print(f"Warning: Tokenizer save failed: {e}")

    full_state_dict = unwrapped_model.state_dict()
    
    # 加入 .clone().detach().cpu() 以断开与 Backbone 内存的共享
    value_head_state_dict = {
        k.replace("value_head.", ""): v.clone().detach().cpu()
        for k, v in full_state_dict.items() 
        if "value_head." in k
    }
    
    token_score_head_state_dict = {
        k.replace("token_score_head.", ""): v.clone().detach().cpu()
        for k, v in full_state_dict.items() 
        if "token_score_head." in k
    }

    # 检查大小 (如果 keys 为空，说明前缀没对上)
    if accelerator.is_main_process:
        print(f"Value Head Keys: {len(value_head_state_dict)} keys found.")
        if len(value_head_state_dict) == 0:
            print("⚠️ CRITICAL WARNING: No keys found for value_head!  Check model structure.")
    
    # 保存
    try:
        torch.save(value_head_state_dict, os.path.join(output_dir, "value_head.bin"))
        torch.save(token_score_head_state_dict, os.path.join(output_dir, "token_score_head.bin"))
        if accelerator.is_main_process: 
            print(f"Custom heads saved to {output_dir}")
    except Exception as e:
        if accelerator.is_main_process: 
            print(f"Error saving custom heads: {e}")


def evaluate(model, dataloader, loss_fn, accelerator:  Accelerator):
    """评估模型（多卡上做全局聚合）"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating",
            disable=not accelerator.is_main_process
        ):
            # Forward pass for chosen
            chosen_outputs = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                response_start_idx=batch.get("chosen_response_start_idx"),
                edit_mask=batch.get("chosen_edit_mask"),
            )

            # Forward pass for rejected
            rejected_outputs = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                response_start_idx=batch.get("rejected_response_start_idx"),
                edit_mask=batch.get("rejected_edit_mask"),
            )

            # Compute loss (本地)
            loss_dict = loss_fn(
                chosen_reward=chosen_outputs["reward"],
                rejected_reward=rejected_outputs["reward"],
                chosen_weighted_score=chosen_outputs["weighted_score"],
                rejected_weighted_score=rejected_outputs["weighted_score"],
            )

            # 将 loss / acc 在所有进程上聚合再取平均
            loss_tensor = loss_dict["loss"].detach()
            acc_tensor = loss_dict["accuracy"].detach()

            gathered_loss = accelerator.gather_for_metrics(loss_tensor)
            gathered_acc = accelerator.gather_for_metrics(acc_tensor)

            batch_loss = gathered_loss.mean().item()
            batch_acc = gathered_acc.mean().item()

            total_loss += batch_loss
            total_accuracy += batch_acc
            num_batches += 1

    return {
        "eval_loss": total_loss / max(num_batches, 1),
        "eval_accuracy": total_accuracy / max(num_batches, 1),
    }


def train(args):
    
    # 1.初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 初始化 wandb
    if args.use_wandb and accelerator.is_main_process: 
        wandb.init(project=args.wandb_project, config=vars(args))

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    train_dataset = EARMDataset(args.train_data, tokenizer, args.max_length)
    train_collator = EARMDataCollator()
    
    # 注意：这里的 batch_size 是 "Per Device Batch Size"
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=4,
    )

    val_dataloader = None
    if args.val_data: 
        val_dataset = EARMDataset(args.val_data, tokenizer, args.max_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size*4,
            shuffle=False,
            collate_fn=train_collator,
            num_workers=4,
        )

    # 初始化模型
    if accelerator.is_main_process:
        print(f"\n加载模型: {args.model_name}")

    model = EditAwareRewardModel(
        model_name_or_path=args.model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        pivot_token_weight=args.pivot_token_weight,
        edit_token_weight=args.edit_token_weight,
        normal_token_weight=args.normal_token_weight,
    )

    # 启用 gradient checkpointing
    if hasattr(model, "backbone") and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "config") and hasattr(model.backbone.config, "use_cache"):
            model.backbone.config.use_cache = False
        if accelerator.is_main_process:
            print("Enabled gradient checkpointing for backbone")

    # 初始化损失函数
    loss_fn = EARMLoss(
        lambda_weighted=args.lambda_weighted,
        mu_margin=args.mu_margin,
        margin=args.margin,
    )

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ==========================================================================
    # 关键修改：先 prepare，再计算步数，最后初始化 scheduler
    # ==========================================================================

    # 1. Prepare 模型、优化器和数据加载器
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # 2.基于 Prepare 后每个 GPU 分到的数据量计算步数
    # math.ceil 确保如果有余数也会算作一步
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) )
    num_training_steps = num_update_steps_per_epoch * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    if accelerator.is_main_process:
        print("\n步数计算详情:")
        print(f"  Per-device batches (len(dl)) :  {len(train_dataloader)}")
        print(f"  Gradient Accumulation steps  : {args.gradient_accumulation_steps}")
        print(f"  Effective steps per epoch    : {num_update_steps_per_epoch}")
        print(f"  Total training steps         : {num_training_steps}")
        print(f"  Warmup steps                 : {num_warmup_steps}")

    # 3.初始化 Scheduler (使用正确的总步数)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4.手动注册 Scheduler (因为没有经过 prepare)
    accelerator.prepare_scheduler(scheduler)

    # ==========================================================================

    # 创建输出目录
    if accelerator.is_main_process: 
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    global_step = 0
    best_accuracy = 0.0

    if accelerator.is_main_process:
        print("\n开始训练...")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):
                # 1.跑 Chosen (正例)
                chosen_outputs = model(
                    input_ids=batch['chosen_input_ids'],
                    attention_mask=batch['chosen_attention_mask'],
                    # --- 新增：必须传下面这两个参数，否则模型无法屏蔽 Source ---
                    response_start_idx=batch['chosen_response_start_idx'],
                    edit_mask=batch['chosen_edit_mask'] 
                    # ----------------------------------------------------
                )

                # 2.跑 Rejected (负例)
                rejected_outputs = model(
                    input_ids=batch['rejected_input_ids'],
                    attention_mask=batch['rejected_attention_mask'],
                    # --- 新增：必须传下面这两个参数 ---
                    response_start_idx=batch['rejected_response_start_idx'],
                    edit_mask=batch['rejected_edit_mask']
                    # ------------------------------
                )

                # 3.算 Loss
                loss_dict = loss_fn(
                    chosen_reward=chosen_outputs['reward'],
                    rejected_reward=rejected_outputs['reward'],
                    chosen_weighted_score=chosen_outputs['weighted_score'],
                    rejected_weighted_score=rejected_outputs['weighted_score']
                )

                loss = loss_dict['loss']

                # Backward
                accelerator.backward(loss)

                
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step() # 这里的 step 现在是基于正确的步数计算的
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0 and accelerator.is_main_process: 
                    log_dict = {
                        "train/loss": loss_dict["loss"].item(),
                        "train/rank_loss": loss_dict["rank_loss"].item(),
                        "train/weighted_rank_loss": loss_dict["weighted_rank_loss"].item(),
                        "train/margin_loss": loss_dict["margin_loss"].item(),
                        "train/accuracy": loss_dict["accuracy"].item(),
                        "train/chosen_reward": loss_dict["chosen_reward_mean"].item(),
                        "train/rejected_reward": loss_dict["rejected_reward_mean"].item(),
                        "train/reward_margin": loss_dict["reward_margin"].item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + num_batches / max(len(train_dataloader), 1),
                    }
                    if args.use_wandb:
                        wandb.log(log_dict, step=global_step)

                    # Evaluation
                if val_dataloader and global_step % args.eval_steps == 0:
                    eval_results = evaluate(model, val_dataloader, loss_fn, accelerator)

                    if accelerator.is_main_process:
                        print(
                            f"\nStep {global_step}:  "
                            f"eval_loss={eval_results['eval_loss']:.4f}, "
                            f"eval_acc={eval_results['eval_accuracy']:.4f}"
                        )

                        if args.use_wandb:
                            wandb.log(
                                {
                                    "eval/loss": eval_results["eval_loss"],
                                    "eval/accuracy": eval_results["eval_accuracy"],
                                },
                                step=global_step,
                            )

                        if eval_results["eval_accuracy"] > best_accuracy:
                            best_accuracy = eval_results["eval_accuracy"]
                            print(f"\n保存最佳模型 (accuracy:  {best_accuracy:.4f})...")
                            save_model(
                                model, tokenizer, accelerator,
                                os.path.join(args.output_dir, "best_model")
                            )

                    model.train()

                # Checkpoint
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    print(f"\n保存 checkpoint-{global_step}...")
                    save_model(
                        model, tokenizer, accelerator,
                        os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    )

            # Metric gathering for progress bar
            with torch.no_grad():
                loss_tensor = loss_dict["loss"].detach()
                acc_tensor = loss_dict["accuracy"].detach()
                margin_tensor = loss_dict["reward_margin"].detach()

                gathered_loss = accelerator.gather_for_metrics(loss_tensor)
                gathered_acc = accelerator.gather_for_metrics(acc_tensor)
                gathered_margin = accelerator.gather_for_metrics(margin_tensor)

                batch_loss = gathered_loss.mean().item()
                batch_acc = gathered_acc.mean().item()
                batch_margin = gathered_margin.mean().item()

            epoch_loss += batch_loss
            epoch_accuracy += batch_acc
            num_batches += 1

            if accelerator.is_main_process:
                progress_bar.set_postfix(
                    {
                        "loss": f"{batch_loss:.4f}",
                        "acc": f"{batch_acc:.4f}",
                        "margin": f"{batch_margin:.4f}",
                    }
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_accuracy = epoch_accuracy / max(num_batches, 1)

        if accelerator.is_main_process:
            print(
                f"\nEpoch {epoch + 1} 完成:  "
                f"avg_loss={avg_loss:.4f}, avg_acc={avg_accuracy:.4f}"
            )

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\n正在保存最终模型...")
        final_output_dir = os.path.join(args.output_dir, "final_model")
        save_model(model, tokenizer, accelerator, final_output_dir)
        print(f"\n✅ 训练完成！完整模型已保存到: {final_output_dir}")

    if args.use_wandb and accelerator.is_main_process: 
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)