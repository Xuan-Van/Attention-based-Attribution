# Installation

## Environment

```bash
conda create -n cite python=3.10 -y
conda activate cite
pip install vllm torch transformers nltk accelerate
```

## Models

- [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Qwen-2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [t5_xxl_true_nli_mixture](https://huggingface.co/google/t5_xxl_true_nli_mixture)
- [deberta-base-long-nli](https://huggingface.co/tasksource/deberta-base-long-nli)

## Datasets

- [ASQA and ELI5](https://huggingface.co/datasets/princeton-nlp/ALCE-data) are from [Enabling Large Language Models to Generate Text with Citations](https://github.com/princeton-nlp/ALCE)
- [2WikiMultiHopQA (2WikiMQA), Natural Questions (NQ), PopQA, and TriviaQA](https://drive.google.com/file/d/1MVkdc4g9_D4REtaBFKeJ9gMun4qzdQtO/view?usp=share_link) are from [InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales](https://github.com/weizhepei/InstructRAG)

# Scripts

## Our method

1. Local model deployment:
```bash
bash vllm.sh model/LLaMA-3.1-8B-Instruct
```
2. Model response obtainment:
```bash
python src/response.py --file data.json --model model/LLaMA-3.1-8B-Instruct
```
3. Attention-based Attribution:
```bash
for method in global local; do
    python src/attention.py \
        --file result/LLaMA-3.1-8B-Instruct.json \
        --model_path model/LLaMA-3.1-8B-Instruct \
        --head_path head/LLaMA.json \
        --method $method
done
```
4. Answer Attribution Evaluation:
```bash
python src/eval.py --file result/LLaMA-3.1-8B-Instruct_LLaMA_global_1.0_None.json --model_path model/t5_xxl_true_nli_mixture
python src/eval.py --file result/LLaMA-3.1-8B-Instruct_LLaMA_local_1.0_None.json --model_path model/t5_xxl_true_nli_mixture
```

## Ablation

1. Layer-wise Ablation:
```bash
for layer in $(seq 0 31); do
    echo "[" > "${layer}.json"
    for head in $(seq 0 31); do
        if [ $head -lt 31 ]; then
            echo "  [$layer, $head]," >> "${layer}.json"
        else
            echo "  [$layer, $head]" >> "${layer}.json"
        fi
    done
    echo "]" >> "${layer}.json"

    for method in global local; do
        python src/attention.py \
            --file result/LLaMA-3.1-8B-Instruct.json \
            --model_path model/LLaMA-3.1-8B-Instruct \
            --head_path "${layer}.json" \
            --method $method
    done

    rm "${layer}.json"
done
```
2. Effect of $\alpha$:
```bash
for method in global local; do
    for alpha in 0 0.5 1.0 1.5 2.0 2.5 3.0; do
        python src/attention.py \
            --file result/LLaMA-3.1-8B-Instruct.json \
            --model_path model/LLaMA-3.1-8B-Instruct \
            --head_path head/LLaMA.json \
            --method $method \
            --alpha $alpha
    done
done
```
3. Effect of Pooling Granularity:
```bash
for method in global local; do
    for k in $(seq 0.1 0.1 1.0); do
        python src/attention.py \
            --file result/LLaMA-3.1-8B-Instruct.json \
            --model_path model/LLaMA-3.1-8B-Instruct \
            --head_path head/LLaMA.json \
            --method $method \
            --top_k $k
    done
done
```

## Baseline

- Vanilla, Summary, and Snippet are from [Enabling Large Language Models to Generate Text with Citations](https://github.com/princeton-nlp/ALCE)
- MIRAGE is from [Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation](https://github.com/Betswish/MIRAGE-reproduce/tree/main/sec5_longQA)
- $\mathrm{C}^2$-Cite is from [$\mathrm{C}^2$-Cite: Contextual-Aware Citation Generation for Attributed Large Language Models](https://github.com/BAI-LAB/c2cite)