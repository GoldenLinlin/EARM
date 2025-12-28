#!/usr/bin/env python3
import sys
sys.path.append('../utils/ChERRANT/modules')
sys.path.append('../utils/ChERRANT')

import os
import json
import argparse
import random
from typing import List, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
from utils.ChERRANT.modules.alignment import read_cilin, read_confusion, AlignmentNoPos
from utils.ChERRANT.modules.merger import Merger
from pypinyin import lazy_pinyin

# Global config set in main()
PAIRS_PER_ITEM: int = 10
SEED: Optional[int] = None
tokenizer = None
alignment = None
merger = None

def generate_edit_mask(source_ids, target_ids, tokenizer, alignment, merger):
    """
    生成编辑掩码：标记哪些 token 位置发生了编辑
    返回: source_edit_mask, target_edit_mask, num_edits
    """
    source_tokens = [tokenizer.decode(tid) for tid in source_ids]
    target_tokens = [tokenizer.decode(tid) for tid in target_ids]

    source_with_pinyin = list(zip(source_tokens, [lazy_pinyin(t) for t in source_tokens]))
    target_with_pinyin = list(zip(target_tokens, [lazy_pinyin(t) for t in target_tokens]))

    try:
        align_objs = alignment(source_with_pinyin, target_with_pinyin, verbose=False)
        edits = merger(align_objs[0], source_with_pinyin, target_with_pinyin, verbose=False)
    except Exception:
        return [0]*len(source_ids), [0]*len(target_ids), 0

    source_edit_mask = [0] * len(source_ids)
    target_edit_mask = [0] * len(target_ids)

    for e in edits:
        try:
            src_start, src_end = e[1], e[2]
            tgt_start, tgt_end = e[3], e[4]
            
            if 0 <= src_start < len(source_ids):
                source_edit_mask[src_start] = 1
                for i in range(src_start + 1, min(src_end, len(source_ids))):
                    source_edit_mask[i] = 2

            if 0 <= tgt_start < len(target_ids):
                target_edit_mask[tgt_start] = 1
                for i in range(tgt_start + 1, min(tgt_end, len(target_ids))):
                    target_edit_mask[i] = 2
        except Exception:
            continue

    return source_edit_mask, target_edit_mask, len(edits)


def process_item(data) -> List[dict]:
    """
    处理单条数据，构建若干 EARM 训练样本
    """
    source = (data.get('ori') or "").strip()
    if not source:
        return []

    # parse answers
    raw_ans = data.get('ans') or []
    ans_list: List[str] = []
    if isinstance(raw_ans, list):
        for a in raw_ans:
            if a is None: continue
            for sub in str(a).split('\t'):
                sub = sub.strip()
                if sub: ans_list.append(sub)
    else:
        for sub in str(raw_ans).split('\t'):
            sub = sub.strip()
            if sub: ans_list.append(sub)

    # parse candidates
    candidates = data.get('tgt') or []
    cleaned = []
    seen = set()
    for c in candidates:
        if c is None: continue
        s = str(c).strip()
        if s == "": continue
        if s in seen: continue
        seen.add(s)
        cleaned.append(s)
    candidates = cleaned

    if len(candidates) < 1:
        return []

    # Tokenize source (保存纯 IDs)
    try:
        source_ids = tokenizer(source, add_special_tokens=False).input_ids
    except Exception:
        return []

    # Pre-tokenize references
    ref_texts = ans_list if ans_list else [source]
    ref_ids_list = []
    for ref in ref_texts:
        try:
            ref_ids = tokenizer(ref, add_special_tokens=False).input_ids
        except Exception:
            ref_ids = []
        ref_ids_list.append(ref_ids)

    # Compute info for existing candidates
    candidate_info = []
    for cand in candidates:
        try:
            cand_ids = tokenizer(cand, add_special_tokens=False).input_ids
        except Exception:
            continue

        distances = []
        for ref_ids in ref_ids_list:
            if not ref_ids:
                distances.append(0)
                continue
            # 临时计算距离 (不保存这次的 mask)
            _, _, num_edits = generate_edit_mask(cand_ids, ref_ids, tokenizer, alignment, merger)
            distances.append(num_edits)

        min_dist = min(distances) if distances else 0

        # 生成相对于 Source 的 Mask (保存这次的 mask)
        try:
            _, cand_mask, num_edits = generate_edit_mask(source_ids, cand_ids, tokenizer, alignment, merger)
        except Exception:
            cand_mask = [0] * len(cand_ids)
            num_edits = 0

        candidate_info.append({
            'text': cand,
            'ids': cand_ids,  # 关键：保存 IDs
            'edit_distances_to_refs': distances,
            'min_edit_distance_to_refs': min_dist,
            'edit_mask': cand_mask, # 关键：保存 Mask
            'num_edits_from_source': num_edits
        })

    if not candidate_info:
        return []

    # ==================================================================================
    # 保留逻辑：Ensure there's at least one candidate equal to a reference.
    # ==================================================================================
    ref_indices = [i for i, c in enumerate(candidate_info) if c['text'] in set(ref_texts)]
    inserted_ref_index = None
    
    if not ref_indices:
        primary_ref = ref_texts[0] if ref_texts else source
        if primary_ref not in [c['text'] for c in candidate_info]:
            try:
                primary_ids = tokenizer(primary_ref, add_special_tokens=False).input_ids
            except Exception:
                primary_ids = []
                
            # compute distances to refs (primary to refs should be 0 for itself)
            distances = []
            for ref_ids in ref_ids_list:
                if not ref_ids:
                    distances.append(0)
                    continue
                # 临时计算距离
                _, _, num_edits = generate_edit_mask(primary_ids, ref_ids, tokenizer, alignment, merger)
                distances.append(num_edits)
                
            min_dist = min(distances) if distances else 0
            
            # 生成相对于 Source 的 Mask
            try:
                _, prim_mask, num_edits = generate_edit_mask(source_ids, primary_ids, tokenizer, alignment, merger)
            except Exception:
                prim_mask = [0] * len(primary_ids)
                num_edits = 0
                
            # append primary ref as a new candidate
            candidate_info.append({
                'text': primary_ref,
                'ids': primary_ids, # 关键：保存 IDs
                'edit_distances_to_refs': distances,
                'min_edit_distance_to_refs': min_dist,
                'edit_mask': prim_mask, # 关键：保存 Mask
                'num_edits_from_source': num_edits
            })
            inserted_ref_index = len(candidate_info) - 1
            ref_indices = [inserted_ref_index]

    # Build all index pairs where min_dist(chosen) < min_dist(rejected)
    n = len(candidate_info)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            di = candidate_info[i]['min_edit_distance_to_refs']
            dj = candidate_info[j]['min_edit_distance_to_refs']
            if di < dj:
                if candidate_info[i]['text'] != candidate_info[j]['text']:
                    pairs.append((i, j))

    if not pairs:
        return []

    # Decide how many pairs to sample
    k = min(len(pairs), PAIRS_PER_ITEM)
    rnd = random.Random(SEED + hash(source)) if SEED is not None else random

    # Ensure at least one pair has chosen in ref_indices.
    ref_pairs = [p for p in pairs if p[0] in ref_indices]
    sampled_pairs = []

    if ref_pairs:
        # choose one ref_pair to include
        chosen_ref_pair = rnd.choice(ref_pairs)
        sampled_pairs.append(chosen_ref_pair)
        # remove chosen_ref_pair from pool
        remaining_pairs = [p for p in pairs if p != chosen_ref_pair]
        remaining_k = k - 1
        if remaining_k > 0 and remaining_pairs:
            if len(remaining_pairs) <= remaining_k:
                sampled_pairs.extend(remaining_pairs)
            else:
                sampled_pairs.extend(rnd.sample(remaining_pairs, remaining_k))
    else:
        # no ref_pairs possible
        if len(pairs) <= k:
            sampled_pairs = pairs.copy()
        else:
            sampled_pairs = rnd.sample(pairs, k)

    entries = []
    # For source_edit_mask (optional info)
    try:
        ref_for_source = ref_ids_list[0] if ref_ids_list else source_ids
        source_edit_mask, _, _ = generate_edit_mask(source_ids, ref_for_source, tokenizer, alignment, merger)
    except Exception:
        source_edit_mask = [0] * len(source_ids)

    for chosen_idx, rejected_idx in sampled_pairs:
        chosen_info = candidate_info[chosen_idx]
        rejected_info = candidate_info[rejected_idx]
        
        entry = {
            'source': source,
            'source_ids': source_ids,
            'source_edit_mask': source_edit_mask,
            'refs': ref_texts,
            'target': ref_texts[0] if ref_texts else source,
            'target_ids_list': ref_ids_list,
            
            # === 核心保存字段 (配合 Dataset 使用) ===
            'chosen': chosen_info['text'],
            'chosen_ids': chosen_info['ids'],       # 纯 Response IDs
            'chosen_edit_mask': chosen_info['edit_mask'], # 对应 Mask
            
            'rejected': rejected_info['text'],
            'rejected_ids': rejected_info['ids'],   # 纯 Response IDs
            'rejected_edit_mask': rejected_info['edit_mask'], # 对应 Mask
            # =====================================
            
            'chosen_min_edit_distance': chosen_info['min_edit_distance_to_refs'],
            'chosen_edit_distances_to_refs': chosen_info['edit_distances_to_refs'],
            'rejected_min_edit_distance': rejected_info['min_edit_distance_to_refs'],
            'rejected_edit_distances_to_refs': rejected_info['edit_distances_to_refs'],
            'edit_distance_diff': rejected_info['min_edit_distance_to_refs'] - chosen_info['min_edit_distance_to_refs'],
            'num_candidates': len(candidates)
        }
        entries.append(entry)

    return entries


def process_data_wrapper(data_list):
    """包装器用于多进程"""
    num_processes = max(1, cpu_count() - 2)
    output_data_list = []
    skipped = 0

    print(f"使用 {num_processes} 个进程处理数据...")
    
    # 如果数据太少就不开多进程了
    if len(data_list) < 100:
        for item in tqdm(data_list):
            res = process_item(item)
            if res: output_data_list.extend(res)
            else: skipped += 1
    else:
        with Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap_unordered(process_item, data_list), total=len(data_list)):
                if not result:
                    skipped += 1
                    continue
                output_data_list.extend(result)

    print(f"跳过数据: {skipped}, 有效条目: {len(output_data_list)}")
    return output_data_list


def merge_json_files(folder_path):
    """完全复用你原本的合并逻辑"""
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json") and f != "merged_result.json"]
    if not json_files: return None

    print(f"找到 {len(json_files)} 个文件，开始合并...")
    merged_data = {}

    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception: continue

        for item in data:
            ori = (item.get("ori") or "").strip()
            if not ori: continue

            raw_ans = item.get("ans") or []
            ans_list = []
            if isinstance(raw_ans, list):
                for a in raw_ans:
                    if a: [ans_list.append(sub.strip()) for sub in str(a).split('\t') if sub.strip()]
            else:
                [ans_list.append(sub.strip()) for sub in str(raw_ans).split('\t') if sub.strip()]

            tgts = [t for t in (item.get("tgt") or []) if t]

            if ori not in merged_data:
                merged_data[ori] = {"ori": ori, "tgt": [], "ans": ans_list}
            merged_data[ori]["tgt"].extend(tgts)
            if ans_list:
                existing = merged_data[ori].get("ans") or []
                for a in ans_list:
                    if a not in existing: existing.append(a)
                merged_data[ori]["ans"] = existing

    # Filter
    for ori_item in merged_data.values():
        ori_len = len(ori_item["ori"])
        unique_tgts = []
        seen = set()
        for t in ori_item["tgt"]:
            t_clean = str(t).strip()
            if not t_clean or t_clean in seen: continue
            if '\n' in t_clean: continue
            # if t_clean and ori_item["ori"] and t_clean[-1] != ori_item["ori"][-1]: continue # 原有的逻辑
            if abs(len(t_clean) - ori_len) > 100: continue
            seen.add(t_clean)
            unique_tgts.append(t_clean)
        ori_item["tgt"] = unique_tgts

    return list(merged_data.values())


def main(args):
    global tokenizer, alignment, merger, PAIRS_PER_ITEM, SEED
    PAIRS_PER_ITEM = args.pairs_per_item
    SEED = args.seed

    print(f"加载 Tokenizer: {args.model_name}")
    # 必须与训练配置一致
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("加载 ChERRANT...")
    semantic_dict, semantic_class = read_cilin()
    confusion_dict = read_confusion()
    alignment = AlignmentNoPos(semantic_dict, confusion_dict)
    merger = Merger(granularity='char')

    if args.merge_folder:
        data_list = merge_json_files(args.merge_folder)
        if not data_list: return
    else:
        with open(args.data_dir, "r", encoding="utf-8") as f:
            data_list = json.load(f)

    if args.test_mode:
        print(f"Test mode: {args.test_samples} samples")
        data_list = data_list[:args.test_samples]

    print("开始处理...")
    if SEED: random.seed(SEED)
    
    output = process_data_wrapper(data_list)
    
    # Filter
    if args.min_edit_distance_diff > 0:
        output = [e for e in output if e['edit_distance_diff'] >= args.min_edit_distance_diff]

    print(f"保存 {len(output)} 条数据到 {args.output_dir}")
    with open(args.output_dir, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Simple version
    simple_path = args.output_dir.replace('.json', '_simple.json')
    simple_output = [{
        'source': e['source'],
        'chosen': e['chosen'],
        'rejected': e['rejected'],
        'diff': e['edit_distance_diff']
    } for e in output]
    with open(simple_path, 'w', encoding='utf-8') as f:
        json.dump(simple_output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--merge_folder", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--min_edit_distance_diff", type=int, default=1)
    parser.add_argument("--pairs_per_item", type=int, default=10)
    parser.add_argument("--test_mode", action='store_true')
    parser.add_argument("--test_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.data_dir is None and args.merge_folder is None:
        parser.error("Need data_dir or merge_folder")
        
    main(args)