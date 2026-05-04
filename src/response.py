import os
import json
import argparse
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer, AutoConfig


def make_doc_prompt(docs):
    """
    构造文档提示文本。

    参数:
        docs (list[dict]): 文档字典列表, 每个字典包含 "title" 和 "text"。

    返回:
        list[str]: 格式化后的文档提示文本列表。
    """
    docs_text = []
    for idx, doc in enumerate(docs):
        text = f"Document [{idx + 1}](Title: {doc['title']}): {doc['text']}"
        docs_text.append(text)

    return docs_text


def construct_model_input(sample, tokenizer, use_demo=True):
    """
    构造模型输入提示文本。

    参数:
        sample (dict): 包含问题、文档和答案的样本字典。
        tokenizer: 模型的分词器，用于应用聊天模板。
        use_demo (bool): 是否在提示中包含示例演示。

    返回:
        str: 格式化后的模型输入提示文本。
    """
    instruction = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."
    demo = {
        "question": "Which is the most rainy place on earth?",
        "answer": "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861 [1].",
        "docs": [
            {
                "title": "Cherrapunji",
                "text": "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in"
            },
            {
                "title": "Cherrapunji",
                "text": "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall"
            },
            {
                "title": "Mawsynram",
                "text": "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"
            },
            {
                "title": "Earth rainfall climatology",
                "text": "Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of"
            },
            {
                "title": "Going to Extremes",
                "text": "in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where \"survival\""
            }
        ]
    }
    prompt = ""

    if use_demo:
        prompt = instruction + f'\nQuestion: {demo["question"]}\n' + "\n".join(make_doc_prompt(demo["docs"])) + f'\nAnswer: {demo["answer"]}'
        prompt += "\n\n"

    prompt += instruction + f'\nQuestion: {sample["question"]}\n' + "\n".join(make_doc_prompt(sample["docs"])) + "\nAnswer:"

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    return prompt


def query_model(port, prompt, stop_token_ids=None):
    """
    向本地模型服务发送请求并获取生成结果。

    参数:
        port (int): 模型服务的端口号。
        prompt (str): 发送给模型的提示文本。
        stop_token_ids (list[int], optional): 生成过程中用于停止的token ID列表。默认为None。

    返回:
        str: 模型生成的文本结果。
    """
    url = "http://localhost:{}/generate".format(port)
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": 2500,
        "temperature": 0,
        "top_p": 1,
        "stop_token_ids": stop_token_ids
    }
    response = requests.post(url, headers=headers, json=data).json()["text"][0]

    return response[len(prompt):]


def main(args):
    # 加载分词器和模型配置
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # 如果分词器没有设置pad_token，使用eos_token
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    # 设置停止token
    if "glm" not in args.model.lower():
        eos_tokens = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
        stop_token_ids = list(set(tokenizer.encode("\n", add_special_tokens=False) + eos_tokens))
    else:
        stop_token_ids = list(set(config.eos_token_id))

    # 加载评估数据
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_length = len(data)
    print(f"加载了 {data_length} 个样本")

    for i in range(data_length):
        data[i]["model_input"] = construct_model_input(data[i], tokenizer)

    # 创建端口列表（从4100开始）
    ports = [i for i in range(4100, 4100 + args.num_port)]

    # 使用线程池并行发送请求（最多32个线程）
    with ThreadPoolExecutor(max_workers=32) as executor:
        result = [executor.submit(query_model, ports[i % len(ports)], data[i]["model_input"], stop_token_ids) for i in range(data_length)]
        for _ in tqdm(as_completed(result), total=len(result)): pass
        responses = [r.result() for r in result]

    # 构造结果数据
    for i in range(data_length):
        data[i]["docs_text"] = make_doc_prompt(data[i]["docs"])
        data[i]["generation"] = responses[i]  # 添加生成结果到样本数据

    # 保存结果到文件
    save_path = f"result/{os.path.basename(args.model)}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于本地模型服务的推理脚本")
    parser.add_argument("--file", type=str, required=True, help="数据文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型名称或路径")
    parser.add_argument("--num_port", type=int, default=4, help="使用的端口数量")

    args = parser.parse_args()
    main(args)