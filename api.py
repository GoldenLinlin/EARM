# Copyright 2024 Bytedance Ltd.and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Optional
from queue import Empty
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ===============================================================
# 配置常量
# ===============================================================
EARM_CKPT_PATH = "/home/work/EARM/best_model_ch"
DEFAULT_DEVICES = [f"cuda:{i}" for i in range(8)]
BATCH_SIZE_PER_GPU = 512
MAX_SEQ_LEN = 2048


# ===============================================================
# 请求/响应模型
# ===============================================================
class InitRequest(BaseModel):
    ckpt_path: Optional[str] = None
    devices: Optional[List[str]] = None


class ScoreRequest(BaseModel):
    source_texts: List[str]
    responses: List[str]


class ScoreResponse(BaseModel):
    scores: List[float]


class StatusResponse(BaseModel):
    status: str
    num_gpus: Optional[int] = None
    devices: Optional[List[str]] = None


# ===============================================================
# EARM 模型定义
# ===============================================================
class EditAwareRewardModel(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path, config=self.config, trust_remote_code=True
        )
        hidden_size = self.config.hidden_size
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        batch_size = input_ids.size(0)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = outputs.last_hidden_state[
            torch.arange(batch_size, device=input_ids.device), seq_lengths
        ]
        return self.value_head(last_hidden).squeeze(-1)


# ===============================================================
# GPU Worker 进程
# ===============================================================
def gpu_worker_process(
    device: str,
    ckpt_path: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_event: mp.Event,
    shutdown_event: mp.Event
):
    """单个 GPU 的 worker 进程"""
    print(f"[{device}] Worker 启动中...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = EditAwareRewardModel(ckpt_path)
    head_path = os.path.join(ckpt_path, "value_head.bin")
    if os.path.exists(head_path):
        model.value_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    else:
        print(f"[{device}] ⚠️ 警告: 没找到 value_head.bin")
    
    model.eval().to(device).bfloat16()
    print(f"[{device}] ✅ 模型加载完成")
    
    ready_event.set()
    
    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=1.0)
        except Empty:
            continue
        
        if task is None:
            break
        
        request_id, source_texts, responses, indices = task
        
        try:
            scores = _compute_scores_on_device(
                model, tokenizer, device, source_texts, responses
            )
            result_queue.put((request_id, indices, scores, None))
        except Exception as e:
            print(f"[{device}] 计算出错: {e}")
            result_queue.put((request_id, indices, [0.0] * len(source_texts), str(e)))
    
    # 清理 GPU 内存
    del model
    torch.cuda.empty_cache()
    print(f"[{device}] Worker 退出，GPU 内存已释放")


@torch.no_grad()
def _compute_scores_on_device(
    model: EditAwareRewardModel,
    tokenizer,
    device: str,
    source_texts: List[str],
    responses: List[str]
) -> List[float]:
    """在指定设备上计算分数"""
    input_ids_list = []
    attention_mask_list = []
    
    for source_text, resp in zip(source_texts, responses):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": source_text}],
            tokenize=True,
            add_generation_prompt=True
        )
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)
        full_ids = prompt_ids + resp_ids + [tokenizer.eos_token_id]
        input_ids_list.append(full_ids)
        attention_mask_list.append([1] * len(full_ids))
    
    all_scores = []
    for i in range(0, len(input_ids_list), BATCH_SIZE_PER_GPU):
        batch_ids = input_ids_list[i:i + BATCH_SIZE_PER_GPU]
        batch_masks = attention_mask_list[i:i + BATCH_SIZE_PER_GPU]
        
        max_len = min(max(len(x) for x in batch_ids), MAX_SEQ_LEN)
        padded_ids = []
        padded_masks = []
        
        for ids, masks in zip(batch_ids, batch_masks):
            curr_len = len(ids)
            if curr_len > max_len:
                padded_ids.append(ids[:max_len])
                padded_masks.append(masks[:max_len])
            else:
                pad_len = max_len - curr_len
                padded_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
                padded_masks.append(masks + [0] * pad_len)
        
        input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(device)
        mask_tensor = torch.tensor(padded_masks, dtype=torch.long).to(device)
        
        rewards = model(input_tensor, mask_tensor)
        if rewards.ndim == 0:
            all_scores.append(rewards.float().item())
        else:
            all_scores.extend(rewards.view(-1).float().cpu().numpy().tolist())
    
    return all_scores


# ===============================================================
# 多 GPU 管理器
# ===============================================================
class MultiGPUManager:
    """管理多个 GPU worker 进程，支持动态加载/卸载"""
    
    def __init__(self):
        self.devices: List[str] = []
        self.num_gpus: int = 0
        self.ckpt_path: str = ""
        
        self.task_queues: Dict[str, mp.Queue] = {}
        self.result_queue: Optional[mp.Queue] = None
        self.workers: Dict[str, mp.Process] = {}
        self.ready_events: Dict[str, mp.Event] = {}
        self.shutdown_event: Optional[mp.Event] = None
        
        self._request_counter = 0
        self._is_initialized = False
        self._lock = threading.Lock()
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def init_models(
        self,
        ckpt_path: str = EARM_CKPT_PATH,
        devices: List[str] = None
    ):
        """初始化并加载模型到所有 GPU"""
        with self._lock:
            if self._is_initialized:
                raise RuntimeError("模型已初始化，请先调用 cleanup")
            
            if devices is None:
                devices = DEFAULT_DEVICES.copy()
            
            self.devices = devices
            self.num_gpus = len(devices)
            self.ckpt_path = ckpt_path
            
            print(f"正在启动 {self.num_gpus} 个 GPU worker...")
            
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            
            self.result_queue = mp.Queue()
            self.shutdown_event = mp.Event()
            
            for device in self.devices:
                task_queue = mp.Queue()
                ready_event = mp.Event()
                
                worker = mp.Process(
                    target=gpu_worker_process,
                    args=(
                        device,
                        self.ckpt_path,
                        task_queue,
                        self.result_queue,
                        ready_event,
                        self.shutdown_event
                    ),
                    daemon=True
                )
                worker.start()
                
                self.task_queues[device] = task_queue
                self.workers[device] = worker
                self.ready_events[device] = ready_event
            
            print("等待所有 worker 就绪...")
            for device, event in self.ready_events.items():
                event.wait(timeout=300)
                if not event.is_set():
                    self._cleanup_internal()
                    raise RuntimeError(f"Worker {device} 启动超时")
            
            self._is_initialized = True
            print(f"✅ 所有 {self.num_gpus} 个 GPU worker 已就绪")
    
    def score_batch(self, source_texts: List[str], responses: List[str]) -> List[float]:
        """分发任务到多个 GPU 并收集结果"""
        if not self._is_initialized:
            raise RuntimeError("模型未初始化，请先调用 /init")
        
        n_samples = len(source_texts)
        if n_samples == 0:
            return []
        
        request_id = self._request_counter
        self._request_counter += 1
        
        samples_per_gpu = (n_samples + self.num_gpus - 1) // self.num_gpus
        tasks_sent = 0
        
        for gpu_idx, device in enumerate(self.devices):
            start_idx = gpu_idx * samples_per_gpu
            end_idx = min(start_idx + samples_per_gpu, n_samples)
            
            if start_idx >= n_samples:
                break
            
            gpu_sources = source_texts[start_idx:end_idx]
            gpu_responses = responses[start_idx:end_idx]
            gpu_indices = list(range(start_idx, end_idx))
            
            task = (request_id, gpu_sources, gpu_responses, gpu_indices)
            self.task_queues[device].put(task)
            tasks_sent += 1
        
        scores = [0.0] * n_samples
        results_received = 0
        
        while results_received < tasks_sent:
            try:
                rid, indices, batch_scores, error = self.result_queue.get(timeout=300)
                if rid != request_id:
                    continue
                
                if error:
                    print(f"Worker 返回错误: {error}")
                
                for idx, score in zip(indices, batch_scores):
                    scores[idx] = score
                
                results_received += 1
            except Empty:
                raise TimeoutError("等待 GPU 结果超时")
        
        return scores
    
    def _cleanup_internal(self):
        """内部清理方法"""
        if self.shutdown_event:
            self.shutdown_event.set()
        
        for device, queue in self.task_queues.items():
            try:
                queue.put(None)
            except:
                pass
        
        for device, worker in self.workers.items():
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        
        self.task_queues.clear()
        self.workers.clear()
        self.ready_events.clear()
        self.result_queue = None
        self.shutdown_event = None
        self._is_initialized = False
    
    def cleanup(self):
        """清理所有资源，释放 GPU 内存"""
        with self._lock:
            if not self._is_initialized:
                return
            
            print("正在关闭 GPU workers 并释放内存...")
            self._cleanup_internal()
            
            # 强制 GC
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print("✅ 所有 GPU workers 已关闭，内存已释放")


# ===============================================================
# FastAPI 应用
# ===============================================================
app = FastAPI(
    title="EARM Reward Model API",
    description="按需加载的多 GPU 并行推理 EARM 奖励模型服务",
    version="2.0.0"
)

gpu_manager = MultiGPUManager()


@app.post("/init", response_model=StatusResponse)
async def init_models(request: InitRequest = None):
    """
    初始化并加载模型到 GPU
    
    可选参数:
    - ckpt_path: 模型路径
    - devices: GPU 设备列表
    """
    try:
        ckpt_path = request.ckpt_path if request and request.ckpt_path else EARM_CKPT_PATH
        devices = request.devices if request and request.devices else DEFAULT_DEVICES
        
        gpu_manager.init_models(ckpt_path=ckpt_path, devices=devices)
        
        return StatusResponse(
            status="initialized",
            num_gpus=gpu_manager.num_gpus,
            devices=gpu_manager.devices
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=ScoreResponse)
async def compute_score(request: ScoreRequest):
    """
    计算 EARM 奖励分数
    
    请求体:
    - source_texts: 原始文本列表
    - responses: 模型回复列表
    """
    if not gpu_manager.is_initialized:
        raise HTTPException(status_code=503, detail="模型未初始化，请先调用 /init")
    
    if len(request.source_texts) != len(request.responses):
        raise HTTPException(
            status_code=400,
            detail="source_texts 和 responses 长度必须相同"
        )
    
    if len(request.source_texts) == 0:
        return ScoreResponse(scores=[])
    
    try:
        scores = gpu_manager.score_batch(request.source_texts, request.responses)
        return ScoreResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup", response_model=StatusResponse)
async def cleanup_models():
    """清理模型，释放 GPU 内存"""
    try:
        gpu_manager.cleanup()
        return StatusResponse(status="cleaned")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """健康检查"""
    if gpu_manager.is_initialized:
        return StatusResponse(
            status="ready",
            num_gpus=gpu_manager.num_gpus,
            devices=gpu_manager.devices
        )
    return StatusResponse(status="idle")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=7788,
        workers=1,
        log_level="info"
    )
