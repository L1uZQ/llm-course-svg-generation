"""
reward.py — SVG Logo Quality Reward Function
=============================================
用于程序化评估生成SVG徽标质量的奖励函数。
每个检查项有明确的理由说明其为何定义了一个"好"徽标。

设计原则:
  - 检查是程序化的、可解释的、可独立计分的
  - 总分为各项加权求和，权重反映重要性
  - 不是所有SVG都是"好"的SVG —— 区分有效性与美学质量
"""

import re
import math
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Tuple, Any


# ─── 配置 ───────────────────────────────────────────────────────────────
VIEWBOX_SIZE = 256
VIEWBOX_MIN = 0
VIEWBOX_MAX = 256
SVG_NS = "http://www.w3.org/2000/svg"

# 合法的SVG元素（矢量图元）
ALLOWED_ELEMENTS = {
    "svg", "g", "defs", "path", "circle", "ellipse", "rect",
    "polygon", "polyline", "line", "text", "linearGradient",
    "radialGradient", "stop", "clipPath", "mask", "filter",
    "feColorMatrix", "feGaussianBlur", "feBlend", "feOffset",
    "feMerge", "feMergeNode", "feFlood", "feComposite",
    "use", "symbol", "pattern", "title", "desc",
}

# 禁止的元素（不安全、不矢量、外部引用）
FORBIDDEN_ELEMENTS = {
    "image", "script", "foreignObject", "iframe", "style",
    "a", "animate", "animateMotion", "animateTransform",
    "set", "switch",
}

# 奖励权重
WEIGHTS = {
    "syntax_validity": 0.20,      # XML语法有效性 —— 基础门槛
    "structure_validity": 0.15,   # 结构完整性 —— 可渲染的基础
    "viewbox_compliance": 0.08,   # 坐标空间合规 —— 统一画布
    "color_palette": 0.10,        # 配色合理性 —— 视觉专业性
    "element_diversity": 0.08,    # 元素多样性 —— 丰富的矢量表达
    "coordinate_bounds": 0.08,    # 坐标约束 —— 元素在画布内
    "complexity_score": 0.06,     # 复杂度 —— 不过简也不过繁
    "keyword_coverage": 0.12,     # 关键词覆盖 —— 符合提示词意图
    "degeneration_penalty": -0.10, # 退化惩罚 —— 拒绝垃圾输出
    "element_count_bonus": 0.03,  # 合理的元素数
}


# ─── 辅助函数 ────────────────────────────────────────────────────────────

def _extract_svg(text: str) -> str:
    """从文本中提取SVG标签内容"""
    # 移除markdown代码块
    text = re.sub(r'```(?:svg|xml|html)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    # 找到svg标签
    match = re.search(r'<svg[\s\S]*?</svg>', text, re.IGNORECASE)
    if match:
        return match.group(0)
    return text.strip()


def _extract_colors(svg_text: str) -> List[str]:
    """从SVG中提取所有颜色值"""
    patterns = [
        r'#[0-9a-fA-F]{3,8}',           # hex: #fff, #aabbcc, #aabbccff
        r'\brgba?\s*\([^)]+\)',          # rgb() / rgba()
        r'\bhsla?\s*\([^)]+\)',          # hsl() / hsla()
        r'\b(white|black|red|green|blue|yellow|cyan|magenta|'
        r'orange|purple|pink|brown|gray|grey|navy|teal|'
        r'lime|maroon|olive|silver|aqua|fuchsia'
        r'|indigo|violet|gold|tan|beige|ivory|cream|coral'
        r'|salmon|turquoise|lavender|mint|plum|khaki|chocolate'
        r'|crimson|azure|snow|honeydew|cornsilk|wheat|seashell'
        r'|linen|bisque|peachpuff|moccasin|papayawhip|blanchedalmond'
        r'|transparent)\b',
    ]
    colors = []
    for pat in patterns:
        colors.extend(re.findall(pat, svg_text, re.IGNORECASE))
    return colors


def _normalize_color(color: str) -> str:
    """标准化颜色为统一格式"""
    color = color.lower().strip()
    # 将3位hex展开为6位
    if re.match(r'^#[0-9a-f]{3}$', color):
        color = '#' + ''.join(c*2 for c in color[1:])
    # 将6位hex转为rgb近似以去重相近色
    if re.match(r'^#[0-9a-f]{6}$', color):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        # 量化到32级以减少相近色
        r, g, b = (r // 32) * 32, (g // 32) * 32, (b // 32) * 32
        return f"#{r:02x}{g:02x}{b:02x}"
    return color


def _extract_numeric_attrs(svg_text: str) -> List[float]:
    """提取SVG中所有数值属性"""
    nums = []
    for attr in ['cx', 'cy', 'x', 'y', 'x1', 'y1', 'x2', 'y2',
                  'r', 'rx', 'ry', 'width', 'height',
                  'stroke-width', 'font-size', 'opacity']:
        pattern = rf'{attr}\s*=\s*"([\d.]+)"'
        for m in re.finditer(pattern, svg_text):
            try:
                nums.append(float(m.group(1)))
            except ValueError:
                pass
    # path d中的坐标
    path_coords = re.findall(r'\b([\d.]+)\s+([\d.]+)', svg_text)
    for x, y in path_coords:
        try:
            nums.append(float(x))
            nums.append(float(y))
        except ValueError:
            pass
    return nums


def _count_elements_by_type(svg_text: str) -> Dict[str, int]:
    """统计各类SVG元素的数量"""
    counts = {}
    for elem in ALLOWED_ELEMENTS:
        pattern = rf'<\s*{elem}[\s>]'
        counts[elem] = len(re.findall(pattern, svg_text, re.IGNORECASE))
    # Total count
    total = len(re.findall(r'<\s*(\w+)', svg_text))
    counts['_total'] = total
    return counts


# ─── 核心评分函数 ────────────────────────────────────────────────────────

KEYWORD_PATTERNS = re.compile(
    r'(circle|circular|round|ring|oval|sphere|disc|medallion|badge'
    r'|square|rectangular|triangle|triangular|diamond|hexagon'
    r'|shield|crest|seal|emblem'
    r'|star|sun|moon|crescent|cloud|wave|zigzag|spiral|arrow|checkmark'
    r'|leaf|flower|tree|plant|sprout|seed|vine|branch|petal|bloom'
    r'|heart|hand|eye|ear|foot|face|head|wing|feather|flame|fire'
    r'|water|drop|droplet|river|ocean|mountain|peak|hill|valley'
    r'|house|home|building|tower|column|pillar|arch|dome|window|door'
    r'|gear|cog|wrench|hammer|key|lock|shield|sword|crown|star'
    r'|rocket|ship|boat|anchor|compass|globe|map|pin|location'
    r'|book|pen|pencil|scroll|document|letter|envelope|chat'
    r'|music|note|eighth-note|staff|clef|microphone|speaker|sound'
    r'|laptop|phone|tablet|screen|monitor|chip|cpu|circuit|code'
    r'|arrow|cursor|pointer|target|crosshair|bullseye|aim'
    r'|nut|bolt|screw|spring|pipe|tube|cable|wire|plug|socket'
    r'|gradient|shadow|glow|outline|stroke|fill|opacity'
    r'|navy|blue|red|green|orange|gold|yellow|white|black|purple'
    r'|teal|pink|brown|gray|silver|bronze|copper|cream|ivory'
    r'|gradient|linear|radial|solid|transparent|opaque'
    r'|border|frame|band|stripe|ribbon|banner|flag|pendant'
    r'|dot|dash|line|curve|arc|loop|knot|tangle|intersect|overlap'
    r'|thin|thick|bold|light|heavy|delicate|chunky|sleek|angular'
    r'|rounded|soft|sharp|smooth|rough|textured|flat|layered'
    r'|minimal|simple|clean|complex|ornate|detailed|intricate'
    r'|modern|vintage|classic|retro|futuristic|organic|geometric'
    r'|abstract|stylized|structured|asymmetric|balanced|centered'
    r'|flowing|sweeping|rising|descending|radiating|spiraling'
    r'|symmetric|symmetrical|balanced|harmonious|cohesive'
    r'|half|quarter|third|split|divided|sectioned|segmented'
    r'|upper|lower|left|right|inner|outer|top|bottom|center'
    r'|positive|negative|space|breathing|breath|pad|padding'
    r'|background|foreground|midground|depth|perspective|layer'
    r'|satellite|orbit|planet|earth|cosmic|galaxy|universe|constellation'
    r'|atom|molecule|cell|microscope|dna|helix|double-helix'
    r'|drop|drip|drop-shaped|teardrop|droplet'
    r'|open|closed|connected|disconnected|overlapping|adjacent'
    r'|aligned|offset|stacked|nested|enclosed|contained'
    r'|inside|outside|around|above|below|beside|between|within'
    r'|tip|point|apex|peak|summit|crown|cap|top|vertex'
    r'|base|bottom|foot|foundation|pedestal|platform|ground'
    r'|tab|notch|indent|groove|slot|cutout|cut-out|carve)',
    re.IGNORECASE,
)


def score_keyword_coverage(prompt: str, svg_text: str) -> Tuple[float, Dict]:
    """
    检查提示词中的关键描述词在SVG中是否有对应元素。
    
    理由: 模型生成的SVG应该忠实反映提示词中的视觉元素。
    一个好的徽标应该将文字描述转化为对应的矢量图形。
    """
    if not prompt:
        return 0.0, {"matched": [], "unmatched": []}

    prompt_lower = prompt.lower()
    svg_lower = svg_text.lower()

    # 从提示词中提取关键描述词
    prompt_keywords = set(KEYWORD_PATTERNS.findall(prompt))

    matched = []
    unmatched = []

    for kw in prompt_keywords:
        # 在SVG中查找关键词
        if kw.lower() in svg_lower:
            matched.append(kw)
        else:
            unmatched.append(kw)

    if not prompt_keywords:
        return 0.5, {"matched": matched, "unmatched": unmatched}

    coverage = len(matched) / max(len(prompt_keywords), 1)
    return coverage, {
        "matched": matched[:10],
        "unmatched": unmatched[:10],
        "total_keywords": len(prompt_keywords),
        "coverage": round(coverage, 3),
    }


def score_syntax_validity(svg_text: str) -> Tuple[float, Dict]:
    """
    检查SVG是否能被XML解析器正确解析。
    
    理由: 无法解析的SVG无法渲染，是完全无效的输出。
    这是最基础的门槛检查。
    """
    svg = _extract_svg(svg_text)
    details = {"parseable": False, "error": None}

    try:
        ET.fromstring(svg)
        details["parseable"] = True
        return 1.0, details
    except ET.ParseError as e:
        details["error"] = str(e)
        # 尝试宽松解析常见问题
        # 1. HTML实体
        try:
            fixed = svg.replace("&", "&amp;")
            fixed = re.sub(r'&amp;(amp|lt|gt|quot|apos);', r'&\1;', fixed)
            ET.fromstring(fixed)
            details["parseable"] = True
            details["fixed_entities"] = True
            return 0.8, details
        except ET.ParseError:
            pass
        return 0.0, details


def score_structure_validity(svg_text: str) -> Tuple[float, Dict]:
    """
    检查SVG结构完整性: xmlns声明、必要属性、禁止元素。
    
    理由: 结构不合规的SVG可能在某些渲染器中无法正常显示。
    xmlns是SVG命名空间标准所必需的。
    """
    svg = _extract_svg(svg_text)
    details = {}
    score = 1.0

    # 检查是否有<svg>标签
    if not re.search(r'<\s*svg[\s>]', svg, re.IGNORECASE):
        details["has_svg_tag"] = False
        return 0.0, details
    details["has_svg_tag"] = True

    # 检查xmlns
    if SVG_NS not in svg:
        details["has_xmlns"] = False
        score -= 0.3
    else:
        details["has_xmlns"] = True

    # 检查viewBox
    viewbox_match = re.search(r'viewBox\s*=\s*"([^"]+)"', svg)
    if viewbox_match:
        vb = viewbox_match.group(1)
        parts = vb.split()
        if len(parts) == 4:
            try:
                vb_values = [float(p) for p in parts]
                if vb_values[2] > 0 and vb_values[3] > 0:
                    details["has_viewbox"] = True
                    details["viewbox"] = vb_values
                else:
                    details["has_viewbox"] = "invalid_size"
                    score -= 0.1
            except ValueError:
                details["has_viewbox"] = "invalid_format"
                score -= 0.1
        else:
            details["has_viewbox"] = "invalid_parts"
            score -= 0.1
    else:
        details["has_viewbox"] = False
        score -= 0.3

    # 检查禁止元素
    forbidden_found = []
    for elem in FORBIDDEN_ELEMENTS:
        if re.search(rf'<\s*{elem}[\s>]', svg, re.IGNORECASE):
            forbidden_found.append(elem)
    details["forbidden_elements"] = forbidden_found
    if forbidden_found:
        score -= 0.2 * len(forbidden_found)

    # 检查是否以</svg>结束
    if not re.search(r'</svg>\s*$', svg.strip(), re.IGNORECASE):
        details["properly_closed"] = False
        score -= 0.1
    else:
        details["properly_closed"] = True

    return max(0.0, score), details


def score_viewbox_compliance(svg_text: str) -> Tuple[float, Dict]:
    """
    检查viewBox是否为0 0 256 256（数据集中统一的标准）。
    
    理由: 数据集中的目标SVG都使用viewBox="0 0 256 256"。
    偏离此标准意味着模型没有学会正确的画布规范。
    """
    svg = _extract_svg(svg_text)
    details = {}

    match = re.search(r'viewBox\s*=\s*"([^"]+)"', svg)
    if not match:
        details["has_viewbox"] = False
        return 0.0, details

    vb = match.group(1).strip()
    parts = vb.split()
    if len(parts) != 4:
        details["viewbox_parts"] = len(parts)
        return 0.0, details

    try:
        x, y, w, h = [float(p) for p in parts]
    except ValueError:
        details["parse_error"] = True
        return 0.0, details

    details["viewbox"] = [x, y, w, h]
    # x,y应为0 且 w,h应为256
    x_ok = abs(x) < 1
    y_ok = abs(y) < 1
    w_ok = abs(w - VIEWBOX_SIZE) <= 5
    h_ok = abs(h - VIEWBOX_SIZE) <= 5

    score = 0.0
    if x_ok and y_ok:
        score += 0.3
    if w_ok:
        score += 0.35
    if h_ok:
        score += 0.35

    details["x_ok"] = x_ok
    details["y_ok"] = y_ok
    details["w_ok"] = w_ok
    details["h_ok"] = h_ok

    return score, details


def score_color_palette(svg_text: str) -> Tuple[float, Dict]:
    """
    评估配色合理性：颜色数量、去重后的有效颜色数。
    
    理由: 
    - 颜色太少(≤1)意味着缺乏视觉层次
    - 颜色太多(>15)意味着杂乱无章
    - 3-10种颜色通常是专业徽标的合理范围
    """
    colors = _extract_colors(svg_text)
    # 过滤掉transparent/white/black作为背景色时的计数
    raw_count = len(colors)
    normalized = set(_normalize_color(c) for c in colors)

    # 排除常见的非设计色
    design_colors = {c for c in normalized
                     if c not in {"#000000", "#ffffff", "transparent",
                                  "#e0e0e0", "#f0f0f0", "#c0c0c0"}}
    design_count = len(design_colors)
    total_unique = len(normalized)

    details = {
        "raw_color_count": raw_count,
        "unique_colors": total_unique,
        "design_colors": design_count,
        "sample_colors": list(design_colors)[:8],
    }

    # 评分逻辑: 理想范围3-10种设计色
    if design_count == 0:
        return 0.1, details
    elif design_count == 1:
        return 0.3, details
    elif design_count == 2:
        return 0.6, details
    elif 3 <= design_count <= 8:
        return 1.0, details
    elif 9 <= design_count <= 12:
        return 0.8, details
    elif 13 <= design_count <= 18:
        return 0.5, details
    else:  # >18
        return 0.2, details


def score_element_diversity(svg_text: str) -> Tuple[float, Dict]:
    """
    评估SVG元素类型的多样性。
    
    理由: 
    - 只使用一种元素类型（如全是<circle>）意味着模型没有学会利用
      SVG的多元素表达能力
    - 合理使用多种元素（path, circle, rect, polygon等）体现丰富的矢量表达能力
    """
    counts = _count_elements_by_type(svg_text)
    # 排除容器元素
    structural = {"svg", "g", "defs", "title", "desc"}
    draw_types = {k: v for k, v in counts.items()
                  if k not in structural and v > 0 and k != "_total"}

    num_types = len(draw_types)
    total_draw = sum(draw_types.values())

    details = {
        "num_element_types": num_types,
        "element_types": draw_types,
        "total_draw_elements": total_draw,
    }

    if num_types == 0:
        return 0.0, details
    elif num_types == 1:
        return 0.2, details
    elif num_types == 2:
        return 0.5, details
    elif num_types == 3:
        return 0.7, details
    elif num_types == 4:
        return 0.85, details
    elif num_types >= 5:
        return 1.0, details
    return 0.0, details


def score_coordinate_bounds(svg_text: str) -> Tuple[float, Dict]:
    """
    检查坐标是否在viewBox范围内。
    
    理由: 
    - 坐标严重越界意味着元素不可见（渲染在画布外）
    - 部分越界可以接受（裁剪效果），但严重越界说明模型不理解空间约束
    """
    nums = _extract_numeric_attrs(svg_text)
    if not nums:
        return 0.5, {"note": "no numeric attributes found"}

    # 区分坐标型数值和尺寸型数值
    # 坐标型数值一般在0-256范围内；尺寸型可以更大
    in_range = sum(1 for n in nums if VIEWBOX_MIN - 50 <= n <= VIEWBOX_MAX + 50)
    far_out = sum(1 for n in nums if n < -200 or n > 600)

    details = {
        "total_numeric": len(nums),
        "in_range": in_range,
        "far_out_of_bounds": far_out,
        "in_range_ratio": round(in_range / len(nums), 3),
        "sample_outliers": [n for n in nums if n < -100 or n > 500][:5],
    }

    if far_out > len(nums) * 0.3:
        return 0.1, details
    ratio = in_range / max(len(nums), 1)

    if ratio > 0.95:
        return 1.0, details
    elif ratio > 0.85:
        return 0.8, details
    elif ratio > 0.70:
        return 0.6, details
    elif ratio > 0.50:
        return 0.4, details
    else:
        return 0.2, details


def score_complexity(svg_text: str) -> Tuple[float, Dict]:
    """
    评估SVG的复杂度是否合理。
    
    理由:
    - 太短（<100字符）通常意味着退化输出（如只画了一个圆）
    - 太长（>15000字符）可能意味着啰嗦的路径数据或退化
    - 合适的长度范围表明模型产出了有意义的内容
    """
    svg = _extract_svg(svg_text)
    length = len(svg)
    # 统计实际绘制元素
    draw_count = len(re.findall(r'<(circle|ellipse|rect|path|polygon|polyline|line)[\s>]',
                                 svg, re.IGNORECASE))
    # 统计路径命令数量（path的复杂度指标）
    path_commands = len(re.findall(r'\b[MmLlHhVvCcSsQqTtAaZz]\b', svg))

    details = {
        "svg_length": length,
        "draw_elements": draw_count,
        "path_commands": path_commands,
    }

    # 退化检测
    if length < 120:
        return 0.0, details  # 几乎是空的
    elif length < 300:
        return 0.2, details  # 过于简单

    # 长度评分
    if 500 <= length <= 6000:
        len_score = 1.0
    elif 300 <= length < 500:
        len_score = 0.5
    elif 6000 < length <= 10000:
        len_score = 0.7
    elif 10000 < length <= 15000:
        len_score = 0.4
    else:  # >15000
        len_score = 0.2

    # 元素数量评分
    if draw_count >= 8:
        elem_score = 1.0
    elif draw_count >= 5:
        elem_score = 0.7
    elif draw_count >= 3:
        elem_score = 0.4
    else:
        elem_score = 0.1

    # 综合
    score = 0.5 * len_score + 0.5 * elem_score
    return score, details


def score_degeneration(svg_text: str) -> Tuple[float, Dict]:
    """
    检测退化输出模式并扣分。
    
    理由:
    - 退化输出（如重复相同元素、单一填充色块、空SVG）
      表明模型没有真正学会生成有意义的徽标
    - 这是对"钻空子"行为的惩罚
    """
    svg = _extract_svg(svg_text)
    details = {}
    penalty = 0.0

    # 1. 空或几乎空的SVG
    if len(svg) < 150:
        penalty += 0.5
        details["empty_or_near_empty"] = True

    # 2. 全是相同元素的重复
    element_pattern = re.findall(r'<(circle|ellipse|rect|path|polygon|line)[^>]*/>', svg, re.IGNORECASE)
    if len(element_pattern) > 5:
        counter = Counter(element_pattern)
        top_ratio = counter.most_common(1)[0][1] / len(element_pattern)
        if top_ratio > 0.9:
            penalty += 0.3
            details["single_element_type_dominance"] = top_ratio

    # 3. 只包含background rect
    bg_only = re.match(r'^\s*<svg[^>]*>\s*<rect[^>]*fill[^>]*/>\s*</svg>\s*$', svg, re.IGNORECASE)
    if bg_only:
        penalty += 0.7
        details["background_only"] = True

    # 4. 检测文本中是否有非SVG内容的泄露（如markdown、代码注释、prose）
    if re.search(r'```|\*\*|#{1,6}\s', svg):
        penalty += 0.2
        details["markdown_leak"] = True
    if re.search(r'(Here|Sure|Certainly|I\'ll|Let me|Below|Above)', svg, re.IGNORECASE):
        penalty += 0.3
        details["prose_leak"] = True

    # 5. SVG包含明显的JSON或Python代码
    if re.search(r'[{}[\]":,]', svg) and '<svg' not in svg[:20].lower():
        penalty += 0.5
        details["non_svg_content"] = True

    score = max(0.0, 1.0 - penalty)
    return score, details


def score_element_count(svg_text: str) -> Tuple[float, Dict]:
    """
    评估元素总数的合理性。
    
    理由: 合理的元素数量（10-80个）通常对应一个结构良好的徽标。
    太少(<5)缺乏细节，太多(>150)可能混乱或退化。
    """
    counts = _count_elements_by_type(svg_text)
    total = counts.get("_total", 0)

    details = {"total_elements": total}

    if total < 3:
        return 0.0, details
    elif total < 8:
        return 0.2, details
    elif total < 15:
        return 0.5, details
    elif 15 <= total <= 120:
        return 1.0, details
    elif total <= 200:
        return 0.7, details
    else:
        return 0.3, details


# ─── 主评分函数 ──────────────────────────────────────────────────────────

def score_svg(prompt: str, svg_text: str, verbose: bool = False) -> Dict[str, Any]:
    """
    对给定提示词和生成的SVG计算综合奖励分数。
    
    Args:
        prompt: 用于生成SVG的提示词文本
        svg_text: 模型生成的SVG文本
        verbose: 是否返回详细信息
    
    Returns:
        dict with keys: total_score, subscores, details
    """
    svg = _extract_svg(svg_text)
    if not svg:
        return {
            "total_score": 0.0,
            "subscores": {},
            "details": {"error": "No SVG found in text"},
        }

    subscores = {}
    all_details = {}

    # 依次运行各项检查
    checkers = {
        "syntax_validity": (score_syntax_validity, [svg]),
        "structure_validity": (score_structure_validity, [svg]),
        "viewbox_compliance": (score_viewbox_compliance, [svg]),
        "color_palette": (score_color_palette, [svg]),
        "element_diversity": (score_element_diversity, [svg]),
        "coordinate_bounds": (score_coordinate_bounds, [svg]),
        "complexity_score": (score_complexity, [svg]),
        "keyword_coverage": (score_keyword_coverage, [prompt, svg]),
        "degeneration_penalty": (score_degeneration, [svg]),
        "element_count_bonus": (score_element_count, [svg]),
    }

    for name, (func, args) in checkers.items():
        try:
            score, details = func(*args)
        except Exception as e:
            score = 0.0
            details = {"error": str(e)}
        subscores[name] = round(score, 4)
        all_details[name] = details

    # 计算加权总分
    total = 0.0
    for name, score in subscores.items():
        w = WEIGHTS.get(name, 0.0)
        total += w * score

    total = round(max(0.0, min(1.0, total)), 4)

    result = {
        "total_score": total,
        "subscores": subscores,
    }
    if verbose:
        result["details"] = all_details
        result["weights"] = WEIGHTS

    return result


# ─── 兼容旧接口 ──────────────────────────────────────────────────────────

# 保持与原 student_kit/reward.py 的接口兼容
def evaluate(prompt: str, svg: str) -> float:
    """简单接口: 输入prompt和svg，返回总分"""
    return score_svg(prompt, svg)["total_score"]


def evaluate_detailed(prompt: str, svg: str) -> Dict:
    """详细接口: 返回完整评分分解"""
    return score_svg(prompt, svg, verbose=True)


# ─── 自检 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 一个好的SVG示例
    good_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
    <circle cx="128" cy="128" r="100" fill="#1B3A5C"/>
    <circle cx="128" cy="128" r="85" fill="none" stroke="#F2A93B" stroke-width="3"/>
    <path d="M128 60 L140 100 L180 100 L145 125 L155 165 L128 145 L101 165 L111 125 L76 100 L116 100 Z" fill="#5DA88E"/>
</svg>"""

    bad_svg = """<svg viewBox="0 0 256 256"><circle cx="128" cy="128" r="50"/></svg>"""

    empty_svg = "Sure, here is your logo:"

    print("=" * 60)
    print("好SVG评分:")
    result = score_svg("A circular navy badge with a gold star and teal accents", good_svg, verbose=True)
    print(f"  总分: {result['total_score']}")
    for k, v in result['subscores'].items():
        print(f"    {k}: {v}")
    print()
    print("差SVG评分:")
    result = score_svg("A circular navy badge", bad_svg, verbose=False)
    print(f"  总分: {result['total_score']}")
    print()
    print("空输出评分:")
    result = score_svg("A beautiful logo", empty_svg, verbose=False)
    print(f"  总分: {result['total_score']}")
    print("=" * 60)
