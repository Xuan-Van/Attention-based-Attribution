import os
import gc
import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import numpy as np
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM


def find_subtexts_positions(input_text, sub_texts, tokenizer):
    """
    查找多个子字符串在整个字符串中的token起止位置。

    参数:
        input_text (str): 整个文本字符串。
        sub_texts (list[str]): 需要查找的子字符串列表。
        tokenizer (AutoTokenizer), 分词器。

    返回:
        dict: 键为子字符串，值为其对应的token序列的起止位置元组 (start_token, end_token)。
    """
    result = {}

    # 获取整个输入文本的token到字符的映射
    encoding = tokenizer(input_text, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"]

    for sub_text in sub_texts:
        # 检查sub_text是否是input_text的子串
        if sub_text not in input_text:
            raise ValueError(f"子文本 '{sub_text[:30]}...' 不在整个文本中")

        # 查找字符串位置
        start_char = input_text.find(sub_text)
        end_char = start_char + len(sub_text)

        # 根据字符位置映射到token位置
        start_token, end_token = -1, -1
        for i, (start, end) in enumerate(offset_mapping):
            # 检查token是否与子串的起始位置重叠
            if start_token == -1 and start <= start_char < end:
                start_token = i
            # 检查token是否与子串的结束位置重叠
            if end_token == -1 and start < end_char <= end:
                end_token = i

        # 确保找到了起始和结束token
        if start_token == -1 or end_token == -1:
            raise ValueError(f"没能定位到子文本 '{sub_text[:30]}...' 的token序列位置")

        # 键: 字符串，值: token起止位置
        result[sub_text] = (start_token, end_token)

    return result


def create_multi_head_hook(model, target_heads):
    """
    创建一个钩子函数来捕获指定层和头的注意力分数。

    参数:
        model: 预加载的语言模型。
        target_heads (list): 包含要捕获的注意力头的层和头索引，格式为 [[layer_idx, head_idx], ...]。

    返回:
        hooks (list): 注册的钩子列表，调用 remove() 可以移除。
        attention_scores (dict): 键为 (layer_idx, head_idx)，值为对应的注意力分数张量。
    """
    layer_to_heads = defaultdict(list)
    for layer_idx, head_idx in target_heads:
        layer_to_heads[layer_idx].append(head_idx)

    attention_scores = {}

    def attention_hook_factory(layer_idx, head_indices):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    for head_idx in head_indices:
                        head_attention = attn_weights[0, head_idx, :, :].detach().cpu()
                        attention_scores[(layer_idx, head_idx)] = head_attention
        return hook_fn

    hooks = []
    for layer_idx, head_indices in layer_to_heads.items():
        hook = model.model.layers[layer_idx].self_attn.register_forward_hook(attention_hook_factory(layer_idx, head_indices))
        hooks.append(hook)

    return hooks, attention_scores


def compute_attention_to_docs(attention_scores, sentence_spans, doc_spans, alpha, method, top_k):
    """
    计算每个句子与每个文档之间的注意力分数，并选择显著的文档。

    参数:
        attention_scores (dict): 键为 (layer_idx, head_idx)，值为对应的注意力分数张量。
        sentence_spans (dict): 键为句子文本，值为其token起止位置的元组 (start_token, end_token)。
        doc_spans (dict): 键为文档文本，值为其token起止位置的元组 (start_token, end_token)。
        alpha (float): 用于控制选择显著文档的阈值，默认为1.0（选择高于平均值加上alpha倍标准差的文档）。
        method (str): 选择显著文档的阈值方法，"global" 使用整个矩阵的分数计算全局阈值，"local" 使用当前句子的分数计算局部阈值。
        top_k (float or None): 每个子矩阵中选取的最大token比例。None表示使用max pooling，否则使用top-k mean pooling。

    返回:
        list: 每个元素是一个字典，键为句子文本，值为与该句子显著相关的文档ID列表。
    """
    # 构建一个矩阵，行对应句子，列对应文档，值为该句子与该文档之间的最大注意力分数
    score_matrixs = {}
    for (layer, head), attn in attention_scores.items():
        score_matrix = np.zeros((len(sentence_spans), len(doc_spans)))
        for sent_idx, (sent_text, (sent_start, sent_end)) in enumerate(sentence_spans.items()):
            sent_tokens = list(range(sent_start, sent_end + 1))
            for doc_idx, (doc_text, (doc_start, doc_end)) in enumerate(doc_spans.items()):
                doc_tokens = list(range(doc_start, doc_end + 1))
                sub_attn = attn[np.ix_(sent_tokens, doc_tokens)]

                if top_k is not None and 0 < top_k <= 1:
                    flat_attn = sub_attn.flatten()
                    k = max(1, int(len(flat_attn) * top_k))  # 至少保留1个token
                    top_values, _ = torch.topk(flat_attn, k)
                    score_matrix[sent_idx, doc_idx] = top_values.mean().item()
                else:
                    score_matrix[sent_idx, doc_idx] = sub_attn.max().item()
            score_matrixs[(layer, head)] = score_matrix

    # 平均多个头的分数矩阵，并选择显著的文档
    results = []
    scores = sum(score_matrixs.values()) / len(score_matrixs)
    for sent_idx, sent_text in enumerate(sentence_spans.keys()):
        sent_scores = scores[sent_idx, :]
        if method == "global":
            threshold = scores.mean() + alpha * scores.std()  # 使用整个矩阵的分数来计算全局阈值
        else:
            threshold = sent_scores.mean() + alpha * sent_scores.std()  # 使用当前句子的分数来计算局部阈值

        doc_indices = np.where(scores[sent_idx, :] > threshold)[-1]
        results.append({sent_text: (doc_indices + 1).tolist()})

    return results


def replace_citations(sentence, new_doc_ids):
    """
    替换句子中的引用标记为新的文档ID引用。

    参数:
        sentence (str): 包含旧引用标记的句子。
        new_doc_ids (list): 新的文档ID列表。

    返回:
        str: 替换后的句子。
    """
    # 如果新的文档ID列表为空，移除所有引用标记
    if not new_doc_ids:
        cleaned = re.sub(r"\s*\[\d+\]", "", sentence)
        return cleaned.strip()

    new_citation = "".join(f"[{doc_id}]" for doc_id in new_doc_ids)   # 构建新的引用字符串
    cleaned_sentence = re.sub(r"\s*\[\d+\]", "", sentence).strip()  # 移除句子中所有的旧引用标记

    # 在句子末尾添加新的引用标记，如果句子以标点符号结尾，则在标点前添加
    if cleaned_sentence and cleaned_sentence[-1] in ".!?。！？":
        modified = cleaned_sentence[:-1] + " " + new_citation + cleaned_sentence[-1]
    else:
        modified = cleaned_sentence + " " + new_citation

    return modified


def process_one_sample(sample, model, tokenizer, target_heads, alpha, method, top_k):
    """
    处理单个样本：解析生成文本，查找句子和文档位置，注册钩子捕获注意力分数，计算注意力并选择显著文档，生成修改后的文本。

    参数:
        sample: 包含生成文本、输入文本和文档文本的样本字典。
        model: 预加载的语言模型。
        tokenizer: 预加载的分词器。
        target_heads: 需要捕获的注意力头列表，格式为 [[layer_idx, head_idx], ...]。
        alpha: 用于控制选择显著文档的阈值，默认为1.0（选择高于平均值加上alpha倍标准差的文档）。
        method: 选择显著文档的阈值方法，"global" 使用整个矩阵的分数计算全局阈值，"local" 使用当前句子的分数计算局部阈值。
        top_k: 每个子矩阵中选取的最大token比例。None表示使用max pooling，否则使用top-k mean pooling。

    返回:
        str: 修改后的生成文本。
    """
    generation = sample["generation"]
    model_input = sample["model_input"]
    full_text = model_input + generation

    # 解析句子
    sentences = sent_tokenize(generation)

    # 查找句子和文档的token位置
    sentence_spans = find_subtexts_positions(full_text, sentences, tokenizer)
    doc_spans = find_subtexts_positions(full_text, sample["docs_text"], tokenizer)

    # 注册钩子并推理
    hooks, attention_scores = create_multi_head_hook(model, target_heads)
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        model(**inputs)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 计算注意力并选择
    results = compute_attention_to_docs(attention_scores, sentence_spans, doc_spans, alpha, method, top_k)

    # 生成修改后的文本
    modified_sentences = []
    for item in results:
        for sent_text, selected_docs in item.items():
            modified_sent = replace_citations(sent_text, selected_docs)
            modified_sentences.append(modified_sent)

    del attention_scores, hooks, inputs
    gc.collect()
    torch.cuda.empty_cache()

    return " ".join(modified_sentences)


def main(args):
    # 加载数据
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(args.head_path, "r") as f:
        target_heads = json.load(f)
    print(f"加载了 {len(data)} 个样本和 {len(target_heads)} 个目标注意力头")

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="auto",
        num_hidden_layers=max(layer for layer, head in target_heads) + 1
    ).eval()

    # 批量处理
    results = []
    failed_ids = []

    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        try:
            modified_generation = process_one_sample(sample, model, tokenizer, target_heads, args.alpha, args.method, args.top_k)
        except Exception as e:
            print(f"处理样本 {idx} 失败: {e}")
            failed_ids.append(sample["id"])
            modified_generation = sample["generation"]

        # 保留原始所有字段，只替换 generation
        result_item = sample.copy()
        result_item["generation"] = modified_generation
        results.append(result_item)

    # 输出文件
    output_path = args.file.replace(".json", f"_{os.path.basename(args.head_path).replace(".json", "")}_{args.method}_{args.alpha}_{args.top_k}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if failed_ids:
        print(f"失败样本ID: {failed_ids}")
    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("基于注意力分数替换生成文本中的文档引用")
    parser.add_argument("--file", type=str, required=True, help="数据文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--head_path", type=str, required=True, help="目标注意力头的 JSON 文件，格式为 [[layer, head], ...]")
    parser.add_argument("--method", type=str, choices=["global", "local"], help="选择显著文档的阈值方法")
    parser.add_argument("--alpha", type=float, default=1.0, help="控制选择显著文档的阈值")
    parser.add_argument("--top_k", type=float, default=None, help="每个子矩阵中选取的最大token比例。None表示使用max pooling，否则使用top-k mean pooling")

    args = parser.parse_args()

    # 验证 top_k 参数
    if args.top_k is not None and not (0 < args.top_k <= 1):
        raise ValueError("top_k 必须在 0 到 1 之间")

    main(args)