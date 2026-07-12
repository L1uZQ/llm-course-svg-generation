"""
inference.py — 用微调后的模型生成 SVG 徽标并在浏览器中预览

用法:
    python inference.py "A circular navy badge with golden star..." --open
    python inference.py --interactive
    python inference.py "prompt" -t 0 -o logo

建议:
    -t 0   确定性模式，SVG 结构最完整
    -t 0.8 更多样但偶尔会有 XML 错误
"""

import argparse
import sys
import re
import os
import html as html_mod
import tempfile
import webbrowser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = """You are an expert logo designer working in clean, scalable vector graphics. Given a description of a logo's visual elements, output ONE complete SVG document for the logo.

Rules:
- Output ONLY the SVG: a single <svg ...>...</svg> element with an xmlns and viewBox="0 0 256 256". No prose, no markdown, no code fences.
- Compose centered, content roughly within 16..240. Use a small cohesive palette.
- Put gradients/filters in <defs>; use vector primitives only (<path>, <circle>, <ellipse>, <rect>, <polygon>, <line>, <g>). No <image>, external refs, or scripts.
- Draw exactly what the description specifies."""


def load_model(model_dir="gemma3-270m", adapter_dir="adapter"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_dir}...")
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def build_prompt(user_text: str) -> str:
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n{user_text}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def generate(model, tokenizer, prompt_text: str,
             max_tokens=2048, temperature=0.8, top_p=0.95):
    inp = tokenizer(build_prompt(prompt_text), return_tensors="pt",
                    add_special_tokens=False).to(model.device)

    kwargs = {"max_new_tokens": max_tokens,
              "pad_token_id": tokenizer.pad_token_id,
              "eos_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        kwargs.update({"temperature": temperature, "do_sample": True,
                       "top_p": top_p, "top_k": 50})
    else:
        kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inp, **kwargs)
    return tokenizer.decode(
        out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_svg(text: str) -> str:
    """从生成文本中提取 SVG"""
    m = re.search(r'<svg[\s\S]*?</svg>', text, re.IGNORECASE)
    if m:
        return m.group(0)
    if re.search(r'</svg>\s*$', text.strip(), re.IGNORECASE):
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">\n' + text.strip()
    # 提取矢量元素
    cleaned = re.sub(r'^.*?(?=<(circle|rect|path|g|defs|line|polygon|ellipse))',
                     '', text.strip(), count=1, flags=re.IGNORECASE | re.DOTALL)
    if cleaned:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">\n' + cleaned + '\n</svg>'
    return text.strip()


def save_html(svg_text: str, path: str):
    """保存为 HTML 页面，浏览器中同时显示渲染结果和源码"""
    escaped = html_mod.escape(svg_text)
    html_page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SVG Logo Preview</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 20px auto; padding: 0 20px; }}
  h2 {{ margin-bottom: 4px; }}
  .info {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
  .preview {{ border: 2px solid #ddd; border-radius: 10px; padding: 30px;
              text-align: center; background: #fafafa; min-height: 200px; }}
  .preview svg {{ max-width: 320px; max-height: 320px; }}
  .error {{ color: #c0392b; font-size: 13px; margin-top: 8px; }}
  details {{ margin-top: 20px; }}
  summary {{ cursor: pointer; color: #555; font-size: 14px; }}
  .source {{ background: #2d2d2d; color: #f8f8f2; border-radius: 6px; padding: 14px;
             font-size: 12px; max-height: 400px; overflow: auto; white-space: pre-wrap;
             font-family: 'Cascadia Code', 'Fira Code', monospace; line-height: 1.5; }}
</style>
</head>
<body>
<h2>Logo Preview</h2>
<div class="info">{len(svg_text)} chars</div>

<div class="preview">
{svg_text}
</div>
<div class="error" id="err"></div>

<details>
<summary>View Source</summary>
<div class="source">{escaped}</div>
</details>

<script>
window.addEventListener('error', function(e) {{
  if (e.target && e.target.tagName === 'svg') return;
  document.getElementById('err').textContent =
    'Rendering error: ' + (e.message || 'SVG parse failed');
}}, true);
</script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_page)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="用微调模型生成 SVG 徽标")
    p.add_argument("prompt", nargs="?", help="提示词文本")
    p.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    p.add_argument("-o", "--output", type=str, help="输出文件（自动加 .html 后缀）")
    p.add_argument("-t", "--temperature", type=float, default=0.8,
                   help="温度 (0=确定性, 0.8=默认, 1=更多样)")
    p.add_argument("--max-tokens", type=int, default=2048, help="最大 token 数")
    p.add_argument("--open", action="store_true", help="生成后浏览器预览")
    p.add_argument("--model-dir", default="gemma3-270m")
    p.add_argument("--adapter-dir", default="adapter")
    args = p.parse_args()

    model, tokenizer = load_model(args.model_dir, args.adapter_dir)

    # 交互模式
    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Enter prompt, 'quit' to exit, 'save name' to save last result\n")
        last = ""
        while True:
            try:
                u = input("Prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not u:
                continue
            if u.lower() == "quit":
                break
            if u.lower().startswith("save "):
                fn = u[5:].strip()
                if not fn.endswith(".html"):
                    fn += ".html"
                if last:
                    save_html(last, fn)
                    print(f"Saved to {fn}")
                    if args.open:
                        webbrowser.open(os.path.abspath(fn))
                else:
                    print("Nothing to save yet.")
                continue

            print("Generating...")
            raw = generate(model, tokenizer, u,
                           max_tokens=args.max_tokens,
                           temperature=args.temperature)
            svg = extract_svg(raw)
            last = svg
            print(f"({len(svg)} chars)")
            print(svg[:500] + ("..." if len(svg) > 500 else ""))
            print()
        return

    # 单次生成
    if not args.prompt:
        p.print_help()
        return

    raw = generate(model, tokenizer, args.prompt,
                   max_tokens=args.max_tokens,
                   temperature=args.temperature)
    svg = extract_svg(raw)

    if not args.output:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        tmp.close()
        args.output = tmp.name
    elif not args.output.endswith(".html"):
        args.output = args.output.rsplit(".", 1)[0] + ".html"

    save_html(svg, args.output)
    print(f"Saved to {args.output} ({len(svg)} chars)")

    if args.open:
        webbrowser.open(os.path.abspath(args.output))


if __name__ == "__main__":
    main()
