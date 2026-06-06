"""
将 LLM 输出的 Markdown 转为游客易读的纯文本，并生成适合语音朗读的文本。
"""

import re


def markdown_to_plaintext(text: str) -> str:
    """
    去除 Markdown 标记，保留可读中文内容。
    列表转为「1. 2. 3.」数字编号，每行一项。
    """
    if not text:
        return text

    s = text.strip()

    # 代码块、行内代码
    s = re.sub(r"```[\s\S]*?```", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # 标题 # ## ###
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)

    # 链接 [文字](url)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # 粗体、斜体（多轮以处理嵌套残留）
    for _ in range(3):
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
        s = re.sub(r"\*([^*\n]+)\*", r"\1", s)
        s = re.sub(r"__([^_]+)__", r"\1", s)

    # 无序列表：- item / * item / + item
    lines_out: list[str] = []
    list_index = 0
    for line in s.splitlines():
        raw = line.strip()
        if not raw:
            lines_out.append("")
            list_index = 0
            continue

        m = re.match(r"^[-*+]\s+(.*)$", raw)
        if m:
            list_index += 1
            item = _clean_inline(m.group(1))
            lines_out.append(f"{list_index}. {item}")
            continue

        # 已是数字列表
        m_num = re.match(r"^(\d+)[.)]\s+(.*)$", raw)
        if m_num:
            list_index = int(m_num.group(1))
            lines_out.append(f"{list_index}. {_clean_inline(m_num.group(2))}")
            continue

        list_index = 0
        lines_out.append(_clean_inline(raw))

    s = "\n".join(lines_out)

    # 清理残留符号
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def _clean_inline(text: str) -> str:
    """清理行内残留 Markdown"""
    t = text.strip()
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*+", "", t)
    return t.strip()


def plain_text_for_speech(text: str) -> str:
    """
    语音朗读专用：在纯文本基础上，把换行改为停顿，去掉不适合朗读的字符。
    """
    s = markdown_to_plaintext(text)
    # 列表换行改为顿号/句号，避免 TTS 读「换行」
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return s
    # 短行合并为一段，长回答用句号分隔
    parts = []
    for ln in lines:
        if re.match(r"^\d+\.\s", ln):
            parts.append(ln)
        else:
            parts.append(ln)
    s = "。".join(parts)
    s = re.sub(r"。{2,}", "。", s)
    s = re.sub(r"[#*_`~\[\]|]", "", s)
    return s.strip()
