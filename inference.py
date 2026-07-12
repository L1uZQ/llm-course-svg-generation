"""
inference.py — 用微调后的模型生成 SVG 徽标

用法:
    python inference.py "A circular navy badge with golden star..."
    python inference.py --interactive
    python inference.py --prompt-file prompts.txt
"""

import argparse
import sys
import torch
import tempfile
import webbrowser
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = """You are an expert logo designer working in clean, scalable vector graphics. Given a description of a logo's visual elements, output ONE complete SVG document for the logo.

Rules:
- Output ONLY the SVG: a single <svg ...>...</svg> element with an xmlns and viewBox="0 0 256 256". No prose, no markdown, no code fences.
- Compose centered, content roughly within 16..240. Use a small cohesive palette.
- Put gradients/filters in <defs>; use vector primitives only (<path>, <circle>, <ellipse>, <rect>, <polygon>, <line>, <g>). No <image>, external refs, or scripts.
- Draw exactly what the description specifies."""


def load_model(model_dir: str = "gemma3-270m", adapter_dir: str = "adapter"):
    """加载基座模型 + LoRA 适配器"""
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,  # fp32 更稳定，推理时 270M 模型仅占 ~1GB
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, tokenizer


def build_prompt(user_text: str) -> str:
    """构建 Gemma 3 格式的输入（不含 <bos>，由 tokenizer 自动添加）"""
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n{user_text}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def generate_svg(model, tokenizer, prompt_text: str,
                 max_tokens: int = 2048,
                 temperature: float = 0.8,
                 top_p: float = 0.95) -> str:
    """生成 SVG"""
    input_text = build_prompt(prompt_text)
    # 不自动添加 special tokens（prompt 已包含格式控制 token）
    inputs = tokenizer(input_text, return_tensors="pt",
                       add_special_tokens=False).to(model.device)

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update({
                "temperature": temperature,
                "do_sample": True,
                "top_p": top_p,
                "top_k": 50,
            })
        else:
            generation_kwargs["do_sample"] = False

        outputs = model.generate(**inputs, **generation_kwargs)

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # 清理输出：只保留 SVG 部分
    svg = _extract_svg(generated)
    return svg


def _extract_svg(text: str) -> str:
    """从生成文本中智能提取/修复 SVG"""
    import re

    # 1. 优先匹配完整的 <svg>...</svg>
    match = re.search(r'<svg[\s\S]*?</svg>', text, re.IGNORECASE)
    if match:
        return match.group(0)

    # 2. 有 </svg> 结尾但缺少开头 <svg> — 自动补
    if re.search(r'</svg>\s*$', text.strip(), re.IGNORECASE):
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">\n' + text.strip()

    # 3. 完全没有 svg 标签 — 用一个最小框架包住
    cleaned = text.strip()
    # 去掉非 SVG 的前导文本
    cleaned = re.sub(r'^.*?(?=<(circle|rect|path|g|defs|line|polygon|ellipse))', '', cleaned, count=1, flags=re.IGNORECASE | re.DOTALL)
    if cleaned:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">\n' + cleaned + '\n</svg>'

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="用微调模型生成 SVG 徽标")
    parser.add_argument("prompt", nargs="?",
                        help="提示词文本")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式")
    parser.add_argument("--prompt-file", "-f", type=str,
                        help="从文件读取提示词（每行一条）")
    parser.add_argument("--model-dir", default="gemma3-270m",
                        help="基座模型目录")
    parser.add_argument("--adapter-dir", default="adapter",
                        help="适配器目录")
    parser.add_argument("--output", "-o", type=str,
                        help="输出 SVG 到文件（默认打印到控制台）")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                        help="采样温度 (0=确定性, 1=更多样)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="最大生成 token 数")
    parser.add_argument("--open", action="store_true",
                        help="生成后在浏览器中预览")
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(args.model_dir, args.adapter_dir)

    # 交互模式
    if args.interactive:
        print("\n=== 交互模式 ===")
        print("输入提示词后回车，输入 'quit' 退出，输入 'save filename.svg' 保存上次结果\n")

        last_svg = ""
        while True:
            try:
                user_input = input("Prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break

            if user_input.lower().startswith("save "):
                filename = user_input[5:].strip()
                if last_svg:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(last_svg)
                    print(f"Saved to {filename}")
                else:
                    print("No SVG to save yet.")
                continue

            print("Generating...")
            svg = generate_svg(model, tokenizer, user_input,
                               max_tokens=args.max_tokens,
                               temperature=args.temperature)
            last_svg = svg
            print("-" * 40)
            print(svg[:800] + ("..." if len(svg) > 800 else ""))
            print("-" * 40)
            print(f"({len(svg)} chars)\n")

        return

    # 从文件批量生成
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        svgs = []
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")
            svg = generate_svg(model, tokenizer, prompt,
                               max_tokens=args.max_tokens,
                               temperature=args.temperature)
            svgs.append(svg)

            if args.output:
                out_file = args.output.replace(".svg", f"_{i+1}.svg")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(svg)
                print(f"  -> {out_file}")
            else:
                print(svg[:300] + "...")
        return

    # 单次生成
    if not args.prompt:
        parser.print_help()
        return

    svg = generate_svg(model, tokenizer, args.prompt,
                       max_tokens=args.max_tokens,
                       temperature=args.temperature)

    if not args.output:
        # 自动保存到临时文件以便 --open
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        tmp.write(svg.encode("utf-8"))
        tmp.close()
        args.output = tmp.name
        print(f"Saved to {args.output}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"({len(svg)} chars)")

    if args.open:
        import webbrowser
        webbrowser.open(args.output)
        print(f"Opened in browser")


if __name__ == "__main__":
    main()
