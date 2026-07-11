"""
eval_self.py — Self-evaluation script
======================================
对基座模型和微调后模型分别在 valid.jsonl 上生成SVG并评分，
输出 results.json 用于提交。

用法:
    python student_kit/eval_self.py valid.jsonl [--model-dir gemma3-270m] [--adapter-dir adapter/]
"""
import json
import sys
import argparse
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载本地reward函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from student_kit.reward import score_svg


def load_data(jsonl_path: str):
    """加载JSONL数据，返回 (prompts, targets) 列表"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item["messages"]
            # system + user 作为输入
            prompt = ""
            target_svg = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += msg["content"] + "\n"
                elif msg["role"] == "user":
                    prompt += msg["content"]
                elif msg["role"] == "assistant":
                    target_svg = msg["content"]
            data.append((prompt, target_svg))
    return data


def build_input(system_content: str, user_content: str, tokenizer):
    """构建Gemma 3格式的输入"""
    # Gemma 3没有独立的system角色，将system内容拼入user turn
    return (
        f"<bos><start_of_turn>user\n"
        f"{system_content}\n\n{user_content}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def extract_prompt_from_messages(messages):
    """从消息列表提取系统提示和用户提示"""
    system = ""
    user = ""
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            user = msg["content"]
    return system, user


def generate_svg(model, tokenizer, prompt_text: str, system_text: str,
                 max_new_tokens: int = 2048) -> str:
    """生成SVG"""
    input_text = build_input(system_text, prompt_text, tokenizer)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    return generated.strip()


def generate_svg_deterministic(model, tokenizer, prompt_text: str,
                                system_text: str, max_new_tokens: int = 2048) -> str:
    """用固定解码设置生成SVG（用于公平对比）"""
    input_text = build_input(system_text, prompt_text, tokenizer)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,     # 确定性
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    return generated.strip()


def evaluate_model(model, tokenizer, data, model_name: str,
                   deterministic: bool = True):
    """评估模型在数据集上的表现"""
    results = []
    total_score = 0.0
    total_target_score = 0.0

    for i, (prompt_text, target_svg) in enumerate(data):
        # 从原始数据中获取system提示
        system_text = ("You are an expert logo designer working in clean, "
                       "scalable vector graphics. Given a description of a "
                       "logo's visual elements, output ONE complete SVG document "
                       "for the logo.\n\nRules:\n- Output ONLY the SVG: a single "
                       "<svg ...>...</svg> element with an xmlns and "
                       'viewBox="0 0 256 256". No prose, no markdown, no code '
                       "fences.\n- Compose centered, content roughly within "
                       "16..240. Use a small cohesive palette.\n- Put "
                       "gradients/filters in <defs>; use vector primitives only "
                       "(<path>, <circle>, <ellipse>, <rect>, <polygon>, <line>, "
                       "<g>). No <image>, external refs, or scripts.\n- Draw "
                       "exactly what the description specifies.")

        t0 = time.time()
        if deterministic:
            generated = generate_svg_deterministic(model, tokenizer,
                                                    prompt_text, system_text)
        else:
            generated = generate_svg(model, tokenizer, prompt_text, system_text)
        elapsed = time.time() - t0

        # 评分
        gen_score = score_svg(prompt_text, generated)
        target_score = score_svg(prompt_text, target_svg)

        total_score += gen_score["total_score"]
        total_target_score += target_score["total_score"]

        results.append({
            "index": i,
            "prompt": prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
            "generated_svg": generated[:500] + "..." if len(generated) > 500 else generated,
            "score": gen_score["total_score"],
            "target_score": target_score["total_score"],
            "gen_time_sec": round(elapsed, 2),
        })

        if (i + 1) % 5 == 0:
            print(f"  [{model_name}] {i+1}/{len(data)} done, "
                  f"avg_score={total_score/(i+1):.3f}")

    avg_score = total_score / len(data) if data else 0.0
    avg_target = total_target_score / len(data) if data else 0.0

    return {
        "model": model_name,
        "num_samples": len(data),
        "average_score": round(avg_score, 4),
        "average_target_score": round(avg_target, 4),
        "total_score": round(total_score, 4),
        "samples": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-evaluate base vs fine-tuned model")
    parser.add_argument("data", help="Path to valid.jsonl")
    parser.add_argument("--model-dir", default="gemma3-270m",
                        help="Path to base Gemma 3 270M model")
    parser.add_argument("--adapter-dir", default="adapter",
                        help="Path to LoRA adapter")
    parser.add_argument("--output", default="results.json",
                        help="Output JSON path")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Skip fine-tuned evaluation (base only)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic decoding (default: True)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Evaluation Script")
    print(f"Data: {args.data}")
    print(f"Model: {args.model_dir}")
    print(f"Adapter: {args.adapter_dir}")
    print(f"Deterministic: {args.deterministic}")
    print("=" * 60)

    # 加载数据
    data = load_data(args.data)
    print(f"\nLoaded {len(data)} validation samples")

    # 加载tokenizer和基座模型
    print("\n[1/2] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()
    print("Base model loaded.")

    # 评估基座模型
    print("\n--- Evaluating Base Model ---")
    base_results = evaluate_model(base_model, tokenizer, data, "base",
                                   deterministic=args.deterministic)
    print(f"Base average score: {base_results['average_score']:.4f}")
    print(f"Target average score: {base_results['average_target_score']:.4f}")
    print(f"Gap to target: {base_results['average_score'] - base_results['average_target_score']:.4f}")

    # 释放基座模型显存
    del base_model
    torch.cuda.empty_cache()

    ft_results = None
    if not args.no_adapter and os.path.exists(args.adapter_dir):
        print("\n[2/2] Loading fine-tuned model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        ft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        ft_model.eval()
        print("Fine-tuned model loaded.")

        print("\n--- Evaluating Fine-tuned Model ---")
        ft_results = evaluate_model(ft_model, tokenizer, data, "fine-tuned",
                                     deterministic=args.deterministic)
        print(f"Fine-tuned average score: {ft_results['average_score']:.4f}")
        improve = ft_results['average_score'] - base_results['average_score']
        print(f"Improvement: {improve:+.4f}")

        del ft_model, base_model
        torch.cuda.empty_cache()
    elif not args.no_adapter:
        print(f"\n[2/2] Adapter not found at {args.adapter_dir}, skipping fine-tuned eval.")

    # 汇总
    output = {
        "base": {
            "average_score": base_results["average_score"],
            "num_samples": base_results["num_samples"],
            "samples": base_results["samples"],
        },
        "reward_function": "student_kit/reward.py",
        "eval_config": {
            "deterministic": args.deterministic,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    if ft_results:
        output["fine_tuned"] = {
            "average_score": ft_results["average_score"],
            "num_samples": ft_results["num_samples"],
            "samples": ft_results["samples"],
            "improvement": round(
                ft_results["average_score"] - base_results["average_score"], 4
            ),
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
