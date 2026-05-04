import re
import json
import copy
import argparse
from tqdm import tqdm

import torch
import numpy as np
from nltk import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def remove_citations(sentence):
    """
    从句子中去除引用标记，返回纯文本句子。

    参数:
        sentence (str): 包含引用标记的句子字符串。

    返回:
        str: 去除引用标记后的纯文本句子。
    """
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sentence)).replace(" |", "").replace("]", "")


def _run_nli_autoais(passage, claim, model, tokenizer):
    """
    使用AutoAIS的NLI模型判断前提（passage）是否蕴含假设（claim）。

    参数:
        passage (str): 联合前提文本，通常是多个文档拼接而成。
        claim (str): 目标句子（去除引用后的文本）。
        model: 预加载的AutoAIS NLI模型。
        tokenizer: 预加载的AutoAIS NLI模型对应的分词器。

    返回:
        int: 1表示前提支持假设（蕴含），0表示不支持。
    """
    input_text = f"premise: {passage} hypothesis: {claim}"

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return 1 if result == "1" else 0


def compute_autoais(data, model, tokenizer, at_most_citations=None):
    """
    使用AutoAIS的NLI模型评估生成文本中每个句子的引用支持情况。

    参数:
        data (list): 包含生成文本和相关文档的样本列表，每个样本是一个字典，包含"generation"和"docs"等字段。
    """

    def _format_document(doc):
        # 得到文档的字符串表示
        return f'Title: {doc["title"]}\n Content: {doc["text"]}'

    # 记录每个样本的AIS得分
    ais_scores = []
    ais_scores_prec = []
    ais_scores_f1 = []
    cite_num_lst = []

    for item in tqdm(data, total=len(data)):
        sents = sent_tokenize(item["generation"])

        if len(sents) == 0: continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0  # 召回率分子（被支持的句子数）
        entail_prec = 0  # 精确率分子（正确且必要的引用数）
        total_citations = 0  # 精确率分母（总引用数）

        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]
            joint_entail = -1  # -1表示未决定，0=不支持，1=支持

            # 提取句子中的引用编号
            ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # 减1转为0-based索引

            if len(ref) == 0:  # 没有引用：直接判定为不蕴含（无法验证来源）
                joint_entail = 0
            elif any([ref_id >= len(item["docs"]) for ref_id in ref]):  # 引用编号超出文档范围：无效引用，判定为不蕴含
                joint_entail = 0
            else:  # 截断引用数量并准备联合前提文本
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = "\n".join([_format_document(item["docs"][psgs_id]) for psgs_id in ref])

            # 如果尚未决定，则运行NLI判断联合文档是否蕴含句子
            if joint_entail == -1:
                joint_entail = _run_nli_autoais(joint_passage, target_sent, model, tokenizer)

            entail += joint_entail

            # 计算精确率：如果句子被支持且引用了多个文档，需要检查每个文档是否必要
            if joint_entail and len(ref) > 1:
                for psgs_id in ref:
                    # 单独使用该文档，看是否蕴含目标句子
                    passage = _format_document(item["docs"][psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent, model, tokenizer)

                    # 如果单独文档不支持：去掉该文档后看是否仍支持
                    if not nli_result:
                        # 复制引用列表并移除当前文档
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = "\n".join([_format_document(item["docs"][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent, model, tokenizer)

                        if nli_result:  # 去掉该文档后仍支持：该文档是不必要的
                            continue
                        else:  # 去掉该文档后不支持：该文档是必要的
                            entail_prec += 1
                    else:  # 单独文档就支持：该文档是必要的
                        entail_prec += 1
            else:  # 如果句子被支持但没有引用或只有一个引用，则该引用被视为必要的
                entail_prec += joint_entail

        citation_recall = entail / len(sents)
        citation_prec = entail_prec / total_citations if total_citations > 0 else 0
        if citation_recall + citation_prec > 0:
            citation_f1 = 2 * citation_recall * citation_prec / (citation_recall + citation_prec)
        else:
            citation_f1 = 0

        ais_scores.append(citation_recall)
        ais_scores_prec.append(citation_prec)
        ais_scores_f1.append(citation_f1)
        cite_num_lst.append(total_citations)

    # 返回平均召回率和平均精确率（转换为百分比）
    return ais_scores, ais_scores_prec, ais_scores_f1, cite_num_lst


def main(args):
    # 加载AutoAIS的NLI模型和分词器
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)  # 使用慢速但更稳定的分词器

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 按dataset字段分组
    dataset_groups = {}
    for i, item in enumerate(data):
        dataset = data[i]["dataset"]
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(item)

    all_ais_scores = []
    all_ais_scores_prec = []
    all_ais_scores_f1 = []
    all_cite_num_lst = []
    results = []

    # 逐个数据集计算AIS得分，并汇总结果
    for dataset, items in dataset_groups.items():
        ais_scores, ais_scores_prec, ais_scores_f1, cite_num_lst = compute_autoais(items, model, tokenizer, at_most_citations=args.at_most_citations)

        all_ais_scores.extend(ais_scores)
        all_ais_scores_prec.extend(ais_scores_prec)
        all_ais_scores_f1.extend(ais_scores_f1)
        all_cite_num_lst.extend(cite_num_lst)

        results.append({
            "dataset": dataset,
            "num_samples": len(items),
            "citation_rec": round(100 * np.mean(ais_scores), 2) if ais_scores else 0,
            "citation_prec": round(100 * np.mean(ais_scores_prec), 2) if ais_scores_prec else 0,
            "citation_f1": round(100 * np.mean(ais_scores_f1), 2) if ais_scores_f1 else 0,
            "avg_citations": round(np.mean(cite_num_lst), 2) if cite_num_lst else 0
        })

    results.append({
        "dataset": "all",
        "num_samples": len(data),
        "citation_rec": round(100 * np.mean(all_ais_scores), 2) if all_ais_scores else 0,
        "citation_prec": round(100 * np.mean(all_ais_scores_prec), 2) if all_ais_scores_prec else 0,
        "citation_f1": round(100 * np.mean(all_ais_scores_f1), 2) if all_ais_scores_f1 else 0,
        "avg_citations": round(np.mean(all_cite_num_lst), 2) if all_cite_num_lst else 0
    })

    output_path = args.file.replace(".json", ".score")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("评估生成文本中每个句子的引用支持情况")
    parser.add_argument("--file", type=str, required=True, help="包含生成文本和相关文档的输入文件")
    parser.add_argument("--model_path", type=str, required=True, help="AutoAIS NLI模型的路径")
    parser.add_argument("--at_most_citations", type=int, default=None, help="每个句子最多考虑的引用数量")

    args = parser.parse_args()
    main(args)