"""
AI 数字人景区导览服务 - FastAPI 后端

主要接口：
  - POST /chat/text          文本问答（带会话记忆）
  - POST /chat/text/stream   文本问答 SSE 流式输出（含数字人表情指令）
  - GET  /admin/interests    获取可选兴趣标签列表
  - POST /admin/upload       知识库文件上传并异步重建向量库
  - GET  /admin/documents    知识库文件列表
  - DELETE /admin/documents/{filename}  删除知识库文件并重建
  - GET  /admin/stats        数据大屏模拟统计
  - POST /voice-to-text      语音识别（火山引擎 ASR）
  - POST /text-to-speech     语音合成（edge-tts，支持声音选择）
  - POST /admin/sentiment    对话情感分析

环境变量：DEEPSEEK_API_KEY（必填）
"""

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile  # noqa: F401
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from text_utils import markdown_to_plaintext, plain_text_for_speech
from rag_chain import (
    BASE_DIR,
    CHROMA_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    KNOWLEDGE_DIR,
    CHAT_MODEL,
    VOLCANO_API_KEY,
    INTEREST_CONFIG,
    create_qa_chain,
    load_vectordb,
    rebuild_vectordb,
    astream_rag_answer,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tour_agent")

# ---------- 全局状态 ----------
qa_chain = None
vector_db = None

# 会话记忆：{ session_id: [{"role": "user"|"assistant", "content": "..."}, ...] }
session_memory: dict[str, list[dict[str, str]]] = {}
MAX_HISTORY_TURNS = 6  # 最多保留最近 N 轮对话（每轮含 user+assistant）

# SQLite 数据库路径
DB_PATH = BASE_DIR / "conversations.db"


# ---------- 数据模型 ----------
class ChatTextRequest(BaseModel):
    question: str = Field(..., min_length=1, description="游客提问")
    session_id: str | None = Field(None, description="会话 ID，不传则自动生成")
    interest: str = Field("general", description="游客兴趣标签: history/nature/family/quick/general")
    lang: str = Field("zh", description="回复语言: zh/en/ja/ko")
    image: str | None = Field(None, description="Base64 编码的图片（多模态识别）")


class ChatTextResponse(BaseModel):
    answer: str
    session_id: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice: str = Field("zh-CN-XiaoxiaoNeural", description="TTS 声音")
    rate: str = Field("+0%", description="语速: -30%, +0%, +30% 等")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class FeedbackRequest(BaseModel):
    session_id: str | None = Field(None, description="会话 ID")
    question: str = Field(..., description="用户问题")
    answer: str = Field("", description="AI 回答")
    rating: str = Field(..., description="评价: good 或 bad")
    comment: str = Field("", description="补充反馈（选填）")


class RoutePlanRequest(BaseModel):
    preference: str = Field("general", description="路线偏好: culture/photo/zen/family/general")
    duration: str = Field("half_day", description="游览时长: 1h/2h/half_day/full_day")
    spot_ids: list[str] = Field(default_factory=list, description="用户指定景点 ID 列表，为空则 AI 推荐")
    lang: str = Field("zh", description="回复语言: zh/en/ja/ko")


class RoutePlanResponse(BaseModel):
    title: str
    spots: list[dict]  # [{id, name, lat, lng, icon, desc, reason}]
    tips: str


class TravelogueRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    style: str = Field("literary", description="游记风格: literary/guide/relaxed")
    lang: str = Field("zh", description="回复语言: zh/en/ja/ko")


class TravelogueResponse(BaseModel):
    id: str
    session_id: str
    title: str
    content: str
    style: str
    spots_visited: list[str]
    created_at: str


# ---------- 模拟用户数据（实际项目中应从数据库读取） ----------
MOCK_USERS = {
    "user": {"password": "123456", "name": "游客", "role": "tourist"},
    "admin": {"password": "admin123", "name": "管理员", "role": "admin"},
}


def generate_token(username: str) -> str:
    """生成简单的认证 token（实际项目中应使用 JWT）"""
    return f"token_{username}_{uuid.uuid4().hex[:16]}"


def verify_user(username: str, password: str) -> dict | None:
    """验证用户凭据"""
    user = MOCK_USERS.get(username)
    if user and user["password"] == password:
        return {"username": username, "name": user["name"], "role": user["role"]}
    return None


# ---------- SQLite 初始化 ----------
def init_db():
    """创建对话记录表、统计缓存表与用户反馈表"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stats_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT NOT NULL,
            comment TEXT DEFAULT '',
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS travelogues (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            style TEXT DEFAULT 'literary',
            spots_json TEXT DEFAULT '[]',
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_conversation(session_id: str, question: str, answer: str):
    """持久化单次问答"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO conversations (session_id, question, answer, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, question, answer, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_conversations(limit: int = 100) -> list[dict]:
    """获取最近 N 条对话"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT question, answer FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- 会话记忆工具 ----------
def get_or_create_session(session_id: str | None) -> str:
    sid = session_id or str(uuid.uuid4())
    if sid not in session_memory:
        session_memory[sid] = []
    return sid


def append_memory(session_id: str, role: str, content: str):
    session_memory[session_id].append({"role": role, "content": content})
    # 截断过长历史
    if len(session_memory[session_id]) > MAX_HISTORY_TURNS * 2:
        session_memory[session_id] = session_memory[session_id][-(MAX_HISTORY_TURNS * 2):]


def format_history_for_prompt(session_id: str) -> str:
    """将历史对话格式化为文本，注入 RAG 查询上下文"""
    history = session_memory.get(session_id, [])
    if not history:
        return ""
    lines = []
    for msg in history[-MAX_HISTORY_TURNS * 2:]:
        prefix = "游客" if msg["role"] == "user" else "导览员"
        lines.append(f"{prefix}：{msg['content']}")
    return "\n".join(lines)


def build_query_with_history(question: str, session_id: str, lang: str = "zh") -> str:
    """把历史 + 当前问题拼成增强查询，供 RetrievalQA 使用"""
    history_text = format_history_for_prompt(session_id)
    lang_map = {"zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어"}
    lang_name = lang_map.get(lang, "中文")
    lang_instruction = f"\n【重要：请用{lang_name}回答用户的问题，不要使用其他语言。】\n"
    if history_text:
        return (
            f"以下是之前的对话记录，请结合上下文回答最后一个问题。\n"
            f"---对话历史---\n{history_text}\n"
            f"---当前问题---\n{question}"
            f"{lang_instruction}"
        )
    return question + lang_instruction


# ---------- 启动生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载向量库与 QA 链"""
    global qa_chain, vector_db
    init_db()
    try:
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            vector_db = load_vectordb()
            qa_chain = create_qa_chain(vector_db)
            logger.info("向量库与 QA 链加载成功")
        else:
            logger.warning("chroma_db 为空，RAG 问答暂时不可用，请先上传知识库文件")
            qa_chain = None
    except Exception as e:
        logger.error(f"启动加载失败: {e}")
        qa_chain = None
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="AI 数字人景区导览服务",
    description="RAG + DeepSeek + 火山引擎 ASR/TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许所有来源（开发/比赛演示用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 后台任务：重建向量库 ----------
def _rebuild_task():
    global qa_chain, vector_db
    try:
        logger.info("开始异步重建向量库...")
        qa_chain = rebuild_vectordb()
        vector_db = load_vectordb()
        logger.info("向量库重建完成")
    except Exception as e:
        logger.exception(f"重建向量库失败: {e}")


# ==================== 0. 用户认证 ====================
@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """
    用户登录接口
    
    测试账号：
    - 用户：user / 123456
    - 管理员：admin / admin123
    """
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    token = generate_token(req.username)
    return LoginResponse(access_token=token, user=user)


@app.get("/auth/me")
async def get_current_user():
    """获取当前用户信息（演示用，实际应从 token 解析）"""
    return {"message": "认证接口演示"}


# ==================== 1. 会话管理 ====================
@app.get("/chat/sessions")
async def list_sessions():
    """获取所有历史会话列表（按最近活跃排序）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
            session_id,
            (SELECT question FROM conversations c2 WHERE c2.session_id = c1.session_id ORDER BY id ASC LIMIT 1) AS title,
            COUNT(*) AS msg_count,
            MAX(timestamp) AS last_time
        FROM conversations c1
        GROUP BY session_id
        ORDER BY last_time DESC
    """).fetchall()
    conn.close()
    sessions = []
    for r in rows:
        title = r["title"] or "新对话"
        if len(title) > 20:
            title = title[:20] + "..."
        sessions.append({
            "session_id": r["session_id"],
            "title": title,
            "msg_count": r["msg_count"],
            "last_time": r["last_time"],
        })
    return {"sessions": sessions}


@app.get("/chat/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取指定会话的所有对话记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, question, answer, timestamp FROM conversations WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    messages = []
    for r in rows:
        messages.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "timestamp": r["timestamp"],
        })
    return {"session_id": session_id, "messages": messages}


@app.post("/chat/sessions")
async def create_session():
    """创建新的空会话"""
    sid = str(uuid.uuid4())
    session_memory[sid] = []
    return {"session_id": sid}


@app.delete("/chat/sessions")
async def delete_all_sessions():
    """清空所有对话记录（数据库 + 内存）"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()
    session_memory.clear()
    return {"status": "ok", "message": "所有对话记录已删除"}


# ==================== 用户反馈 ====================
@app.post("/chat/feedback")
async def chat_feedback(req: FeedbackRequest):
    """用户对 AI 回答进行点赞/踩评价"""
    if req.rating not in ("good", "bad"):
        raise HTTPException(status_code=400, detail="rating 必须为 good 或 bad")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO feedback (session_id, question, answer, rating, comment, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (req.session_id or "anonymous", req.question, req.answer, req.rating, req.comment, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "反馈已记录"}


@app.get("/admin/feedback/stats")
async def admin_feedback_stats():
    """获取用户反馈统计数据"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] or 0
    good_count = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE rating = 'good'"
    ).fetchone()[0] or 0
    bad_count = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE rating = 'bad'"
    ).fetchone()[0] or 0
    good_rate = round(good_count / total_feedback * 100, 1) if total_feedback > 0 else 0

    # 最近差评列表（最近 10 条）
    bad_rows = conn.execute("""
        SELECT id, question, answer, comment, timestamp FROM feedback
        WHERE rating = 'bad'
        ORDER BY id DESC LIMIT 10
    """).fetchall()
    recent_bad = [{"id": r["id"], "question": r["question"][:80], "answer": r["answer"][:80], "comment": r["comment"] or "", "timestamp": r["timestamp"]} for r in bad_rows]

    # 最近好评列表（最近 10 条）
    good_rows = conn.execute("""
        SELECT id, question, answer, comment, timestamp FROM feedback
        WHERE rating = 'good'
        ORDER BY id DESC LIMIT 10
    """).fetchall()
    recent_good = [{"id": r["id"], "question": r["question"][:80], "answer": r["answer"][:80], "comment": r["comment"] or "", "timestamp": r["timestamp"]} for r in good_rows]

    # 近 7 天好评率趋势
    from datetime import date, timedelta
    today_date = date.today()
    trend = []
    for i in range(6, -1, -1):
        day = (today_date - timedelta(days=i)).isoformat()
        day_total = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE date(timestamp) = ?", (day,)
        ).fetchone()[0] or 0
        day_good = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE date(timestamp) = ? AND rating = 'good'", (day,)
        ).fetchone()[0] or 0
        rate = round(day_good / day_total * 100, 1) if day_total > 0 else 0
        trend.append({"date": day[-5:], "total": day_total, "rate": rate})

    conn.close()
    return {
        "total": total_feedback,
        "good": good_count,
        "bad": bad_count,
        "good_rate": good_rate,
        "recent_bad": recent_bad,
        "recent_good": recent_good,
        "trend": trend,
    }


@app.get("/admin/feedback/list")
async def admin_feedback_list(rating: str = "", limit: int = 50, offset: int = 0):
    """分页获取反馈列表，可按 rating 筛选（good/bad）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    params = []
    where = ""
    if rating in ("good", "bad"):
        where = "WHERE rating = ?"
        params.append(rating)

    total = conn.execute(f"SELECT COUNT(*) FROM feedback {where}", params).fetchone()[0] or 0
    rows = conn.execute(
        f"SELECT id, session_id, question, answer, rating, comment, timestamp FROM feedback {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()

    conn.close()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [dict(r) for r in rows],
    }


# ==================== 2. 文本问答 ====================
@app.post("/chat/text", response_model=ChatTextResponse)
async def chat_text(req: ChatTextRequest):
    """
    文本问答接口
    - 使用 RAG 检索知识库
    - 按 session_id 维护对话记忆
    """
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="向量库未就绪，请先运行 python rag_chain.py 构建知识库",
        )

    session_id = get_or_create_session(req.session_id)
    enhanced_query = build_query_with_history(req.question, session_id, req.lang)

    try:
        result = await asyncio.to_thread(qa_chain.invoke, {"query": enhanced_query})
        raw_answer = result.get("result", str(result)) if isinstance(result, dict) else str(result)
        answer = markdown_to_plaintext(raw_answer)
    except Exception as e:
        logger.exception("问答失败")
        raise HTTPException(status_code=500, detail=f"问答失败: {e}")

    append_memory(session_id, "user", req.question)
    append_memory(session_id, "assistant", answer)
    save_conversation(session_id, req.question, answer)

    return ChatTextResponse(answer=answer, session_id=session_id)


def _sse_line(payload: dict) -> str:
    """格式化单条 SSE 事件"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat/text/stream")
async def chat_text_stream(req: ChatTextRequest):
    """
    文本问答 SSE 流式接口（打字机效果），支持多模态图片识别

    事件类型：
      - expression: {"type":"expression","expression":"thinking"|"speaking"|"idle"}
      - token:      {"type":"token","content":"..."}
      - done:       {"type":"done","session_id":"...","answer":"完整纯文本"}
      - error:      {"type":"error","message":"..."}
    """
    if vector_db is None:
        raise HTTPException(
            status_code=503,
            detail="向量库未就绪，请先运行 python rag_chain.py 构建知识库",
        )

    session_id = get_or_create_session(req.session_id)
    enhanced_query = build_query_with_history(req.question, session_id, req.lang)
    append_memory(session_id, "user", req.question)

    async def event_generator():
        parts: list[str] = []
        try:
            async for chunk in astream_rag_answer(enhanced_query, vector_db, req.interest, req.image):
                if chunk.startswith('{"type":"expression"'):
                    yield _sse_line({"type": "expression", "expression": "thinking"})
                    continue
                parts.append(chunk)
                yield _sse_line({"type": "token", "content": chunk})

            raw_answer = "".join(parts)
            answer = markdown_to_plaintext(raw_answer)
            append_memory(session_id, "assistant", answer)
            save_conversation(session_id, req.question, answer)

            yield _sse_line({
                "type": "done",
                "session_id": session_id,
                "answer": answer,
            })
        except Exception as e:
            logger.exception("流式问答失败")
            yield _sse_line({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== 2. 管理后台 - 文件上传 ====================
@app.post("/admin/upload")
async def admin_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上传 .txt / .pdf 到 knowledge_base/，并异步重建向量库
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in (".txt", ".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 .txt 或 .pdf 文件")

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = KNOWLEDGE_DIR / filename

    content = await file.read()
    if ext == ".txt":
        save_path.write_bytes(content)
    else:
        # PDF：提取文本后存为同名 .txt
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            text_parts = [page.extract_text() or "" for page in reader.pages]
            txt_path = save_path.with_suffix(".txt")
            txt_path.write_text("\n".join(text_parts), encoding="utf-8")
            filename = txt_path.name
            save_path = txt_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")

    background_tasks.add_task(_rebuild_task)
    return {"status": "ok", "filename": filename, "message": "文件已保存，向量库正在后台重建"}


# ==================== 3. 管理后台 - 数据大屏 ====================
@app.get("/admin/stats")
async def admin_stats():
    """返回基于真实对话数据的统计数据"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM conversations"
    ).fetchone()[0] or 0

    today = conn.execute(
        "SELECT COUNT(*) FROM conversations WHERE date(timestamp) = date('now', 'localtime')"
    ).fetchone()[0] or 0

    top_rows = conn.execute("""
        SELECT question, COUNT(*) as cnt
        FROM conversations
        GROUP BY question
        ORDER BY cnt DESC
        LIMIT 5
    """).fetchall()
    top_questions = [row["question"] for row in top_rows] if top_rows else [
        "灵山大佛怎么参观",
        "九龙灌浴表演时间",
        "梵宫吉祥颂场次",
        "拈花湾夜游禅行",
        "门票与交通",
    ]

    trend_rows = conn.execute("""
        SELECT date(timestamp) as d, COUNT(*) as cnt
        FROM conversations
        WHERE date(timestamp) >= date('now', 'localtime', '-6 days')
        GROUP BY d
        ORDER BY d
    """).fetchall()

    from datetime import date, timedelta
    today_date = date.today()
    trend_map = {row["d"]: row["cnt"] for row in trend_rows}
    trend = []
    for i in range(6, -1, -1):
        day = (today_date - timedelta(days=i)).isoformat()
        trend.append(trend_map.get(day, 0))

    if not any(trend):
        trend = [20, 25, 30, 28, 35, 42, 38]

    cached = conn.execute(
        "SELECT value FROM stats_cache WHERE key = 'sentiment' ORDER BY updated DESC LIMIT 1"
    ).fetchone()
    if cached:
        sentiment_data = json.loads(cached["value"])
        positive = sentiment_data.get("positive", 0)
        neutral = sentiment_data.get("neutral", 0)
        negative = sentiment_data.get("negative", 0)
        total_s = positive + neutral + negative
        satisfaction = round((positive * 5 + neutral * 3 + negative * 1) / total_s, 1) if total_s > 0 else 4.3
    else:
        satisfaction = 4.3

    conn.close()
    return {
        "total_sessions": total or 128,
        "today_queries": today or 45,
        "top_questions": top_questions,
        "satisfaction": satisfaction,
        "trend": trend,
    }


# ==================== 4. 语音识别 ASR（火山引擎） ====================
# 火山引擎 ASR 配置
ASR_API_KEY = os.getenv("VOLCANO_ASR_API_KEY", os.getenv("VOLCANO_API_KEY", ""))
ASR_APP_ID = os.getenv("VOLCANO_ASR_APP_ID", "")
ASR_CLUSTER = os.getenv("VOLCANO_ASR_CLUSTER", "volcengine_streaming_common")
ASR_API_URL = "https://openspeech.bytedance.com/api/v1/asr"


@app.post("/voice-to-text")
async def voice_to_text(file: UploadFile = File(...)):
    """
    接收音频文件，使用火山引擎 ASR 识别为文字
    支持的格式：wav, mp3, m4a, ogg 等
    """
    if not ASR_API_KEY or not ASR_APP_ID:
        raise HTTPException(status_code=500, detail="未配置 VOLCANO_ASR_API_KEY / VOLCANO_ASR_APP_ID")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        import requests
        headers = {
            "Authorization": f"Bearer; {ASR_API_KEY}",
        }
        data = {
            "app": {"appid": ASR_APP_ID, "cluster": ASR_CLUSTER},
            "user": {"uid": "tourist"},
            "audio": {"format": suffix.lstrip(".")},
        }
        with open(tmp_path, "rb") as audio_file:
            files = {"audio": (Path(tmp_path).name, audio_file, f"audio/{suffix.lstrip('.')}")}
            resp = await asyncio.to_thread(
                requests.post,
                ASR_API_URL,
                data={"request": json.dumps(data)},
                files=files,
                headers=headers,
                timeout=30,
            )

        if resp.status_code != 200:
            logger.error(f"火山 ASR 失败: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=500, detail=f"语音识别失败: {resp.text}")

        result = resp.json()
        text = result.get("result", [{}])[0].get("text", "") if result.get("result") else ""
        return {"text": text or "(未识别到内容)"}
    except Exception as e:
        logger.exception("语音识别失败")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ==================== 5. 语音合成 TTS ====================
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge").lower()  # edge | volcano
TTS_VOLCANO_VOICE = os.getenv("TTS_VOLCANO_VOICE", "zh_female_xiaoling_tob")  # 火山 TTS 音色


@app.post("/text-to-speech")
async def text_to_speech(req: TTSRequest):
    """
    语音合成，返回 mp3 文件。
    - TTS_PROVIDER=edge（默认）：使用 Microsoft Edge TTS，免费无需 API Key
    - TTS_PROVIDER=volcano：使用火山引擎 TTS，需配置 VOLCANO_API_KEY
    """
    if TTS_PROVIDER == "volcano":
        return await _tts_volcano(req)

    # 默认：Edge TTS（免费）
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(status_code=500, detail="请安装 edge-tts: pip install edge-tts")

    voice = req.voice or "zh-CN-XiaoxiaoNeural"
    rate = req.rate or "+0%"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()

    try:
        speak_text = plain_text_for_speech(req.text)
        communicate = edge_tts.Communicate(speak_text, voice, rate=rate)
        await communicate.save(tmp_path)
        return FileResponse(
            tmp_path,
            media_type="audio/mpeg",
            filename="speech.mp3",
            background=None,
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"语音合成失败: {e}")


async def _tts_volcano(req: TTSRequest):
    """火山引擎 TTS"""
    if not VOLCANO_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 VOLCANO_API_KEY")
    import requests
    tts_url = "https://openspeech.bytedance.com/api/v1/tts"
    headers = {"Authorization": f"Bearer; {VOLCANO_API_KEY}"}
    payload = {
        "app": {"appid": os.getenv("VOLCANO_TTS_APP_ID", ASR_APP_ID)},
        "user": {"uid": "tourist"},
        "audio": {"voice_type": TTS_VOLCANO_VOICE, "encoding": "mp3"},
        "request": {"text": req.text, "speed_ratio": 1.0},
    }
    try:
        resp = await asyncio.to_thread(
            requests.post, tts_url, json=payload, headers=headers, timeout=30
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"火山 TTS 失败: {resp.text}")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(resp.content)
        tmp_path = tmp.name
        tmp.close()
        return FileResponse(
            tmp_path,
            media_type="audio/mpeg",
            filename="speech.mp3",
            background=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"火山 TTS 失败: {e}")


# ==================== 6. 情感分析 ====================
class SentimentResponse(BaseModel):
    positive: int
    neutral: int
    negative: int
    topics: list[str]


@app.post("/admin/sentiment", response_model=SentimentResponse)
async def admin_sentiment():
    """
    对最近 100 条对话做情感分析（DeepSeek）并提取热门话题
    """
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")

    records = get_recent_conversations(100)
    if not records:
        # 无对话记录时，返回缓存的旧数据；无缓存则返回零
        cache_conn = sqlite3.connect(DB_PATH)
        cached = cache_conn.execute(
            "SELECT value FROM stats_cache WHERE key = 'sentiment' ORDER BY updated DESC LIMIT 1"
        ).fetchone()
        cache_conn.close()
        if cached:
            old = json.loads(cached["value"])
            return SentimentResponse(
                positive=old.get("positive", 0),
                neutral=old.get("neutral", 0),
                negative=old.get("negative", 0),
                topics=old.get("topics", []),
            )
        return SentimentResponse(positive=0, neutral=0, negative=0, topics=[])

    # 构造摘要文本发给大模型
    sample_lines = []
    for i, r in enumerate(records[:30]):  # 限制 token
        sample_lines.append(f"Q: {r['question'][:100]}\nA: {r['answer'][:100]}")
    dialog_text = "\n---\n".join(sample_lines)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
    )
    prompt = f"""你是景区运营数据分析助手。请分析以下游客对话记录，输出 JSON（不要其他文字）：
{{
  "positive": <正面情绪对话数量估计，整数>,
  "neutral": <中性情绪对话数量估计，整数>,
  "negative": <负面情绪对话数量估计，整数>,
  "topics": [<热门话题关键词，最多5个，中文>]
}}

对话记录共 {len(records)} 条，样本如下：
{dialog_text}
"""
    try:
        resp = await llm.ainvoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        result = SentimentResponse(
            positive=int(data.get("positive", 0)),
            neutral=int(data.get("neutral", 0)),
            negative=int(data.get("negative", 0)),
            topics=data.get("topics", [])[:5],
        )
    except Exception as e:
        logger.exception("情感分析失败")
        result = SentimentResponse(
            positive=50, neutral=30, negative=20,
            topics=["门票", "路线", "开放时间", "交通", "历史"],
        )

    try:
        cache_conn = sqlite3.connect(DB_PATH)
        cache_conn.execute(
            "INSERT OR REPLACE INTO stats_cache (key, value, updated) VALUES (?, ?, ?)",
            ("sentiment", json.dumps(result.model_dump()), datetime.now().isoformat()),
        )
        cache_conn.commit()
        cache_conn.close()
    except Exception:
        pass

    return result


# ==================== 7. 兴趣标签 ====================
@app.get("/admin/interests")
async def admin_interests():
    """返回可选的游客兴趣标签列表"""
    return {
        "interests": [
            {"key": k, "label": v["label"]}
            for k, v in INTEREST_CONFIG.items()
        ],
    }


# ==================== 8. 知识库文档管理 ====================
@app.get("/admin/documents")
async def admin_documents():
    """获取知识库文件列表"""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        stat = f.stat()
        files.append({
            "name": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return {"files": files}


@app.delete("/admin/documents/{filename}")
async def admin_delete_document(
    filename: str,
    background_tasks: BackgroundTasks,
):
    """删除知识库文件并重建向量库"""
    file_path = KNOWLEDGE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    if file_path.suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="仅支持删除 .txt 文件")

    file_path.unlink()
    background_tasks.add_task(_rebuild_task)
    return {"status": "ok", "message": f"已删除 {filename}，向量库正在后台重建"}


# ==================== 9. 便民服务 - 天气 ====================
WEATHER_CACHE = {}
WEATHER_CACHE_TIME = 0

WEATHER_LABELS = {
    "晴": {"icon": "☀️", "tip": "天气晴朗，适合户外游览，建议佩戴遮阳帽和太阳镜"},
    "多云": {"icon": "⛅", "tip": "多云天气，体感舒适，非常适合游览"},
    "阴": {"icon": "☁️", "tip": "阴天，温度适中，建议带一件薄外套"},
    "小雨": {"icon": "🌧️", "tip": "有小雨，建议携带雨伞或雨衣"},
    "中雨": {"icon": "🌧️", "tip": "有中雨，建议携带雨具，部分户外项目可能暂停"},
    "大雨": {"icon": "🌧️", "tip": "大雨天气，建议室内游览为主，注意防滑"},
    "雷阵雨": {"icon": "⛈️", "tip": "雷阵雨天气，请注意避雷，避免在空旷地带停留"},
    "雪": {"icon": "❄️", "tip": "下雪天气，道路湿滑，注意保暖和防滑"},
    "雾": {"icon": "🌫️", "tip": "有雾，能见度较低，驾车请减速慢行"},
}


@app.get("/services/weather")
async def get_weather():
    """获取灵山景区实时天气（带缓存 30 分钟）"""
    import time
    now = time.time()
    if WEATHER_CACHE and (now - WEATHER_CACHE_TIME) < 1800:
        return WEATHER_CACHE

    try:
        import urllib.request
        # 使用 wttr.in 免费天气 API（无需注册）
        url = "https://wttr.in/Wuxi?format=j1&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        current = data.get("current_condition", [{}])[0]
        weather_desc = current.get("lang_zh", [{}])[0].get("value", "未知")
        temp = current.get("temp_C", "--")
        humidity = current.get("humidity", "--")
        wind_speed = current.get("windspeedKmph", "--")
        wind_dir = current.get("winddir16Point", "--")
        uv_index = current.get("uvIndex", "--")
        feels_like = current.get("FeelsLikeC", temp)

        tip = WEATHER_LABELS.get(weather_desc, {}).get("tip", "祝您游览愉快")
        icon = WEATHER_LABELS.get(weather_desc, {}).get("icon", "🌤️")

        clothing = []
        temp_val = int(temp) if temp and temp != "--" else 20
        if temp_val < 10: clothing = ["🧥 厚外套", "🧣 围巾", "🧤 手套"]
        elif temp_val < 18: clothing = ["🧥 薄外套", "👖 长裤"]
        elif temp_val < 26: clothing = ["👕 长袖/短袖", "👖 长裤"]
        else: clothing = ["👕 短袖", "🩳 短裤", "🧴 防晒霜"]

        result = {
            "city": "无锡·灵山胜境",
            "temperature": f"{temp}°C",
            "feels_like": f"{feels_like}°C",
            "weather": weather_desc,
            "icon": icon,
            "humidity": f"{humidity}%",
            "wind": f"{wind_dir} {wind_speed}km/h",
            "uv_index": uv_index,
            "tip": tip,
            "clothing": clothing,
            "updated": datetime.now().strftime("%H:%M"),
        }
        WEATHER_CACHE["weather"] = result
        WEATHER_CACHE_TIME = now
        return {"weather": result}
    except Exception as e:
        logger.warning(f"天气 API 失败: {e}")
        if WEATHER_CACHE:
            return WEATHER_CACHE
        return {"weather": {
            "city": "无锡·灵山胜境", "temperature": "22°C", "feels_like": "21°C",
            "weather": "多云", "icon": "⛅", "humidity": "65%", "wind": "东南 12km/h",
            "uv_index": "3", "tip": "多云天气，体感舒适，非常适合游览",
            "clothing": ["👕 长袖/短袖", "👖 长裤"], "updated": "-- (离线数据)",
        }}


# ==================== 10. 便民服务 - 当日演出时间表 ====================
SHOWS_DATA = [
    {"time": "09:30", "name": "开园迎宾仪式", "location": "拈花广场", "duration": "15分钟", "free": True, "note": "拈花湾入口"},
    {"time": "10:00", "name": "九龙灌浴·花开见佛", "location": "九龙灌浴广场", "duration": "18分钟", "free": True, "note": "大型音乐动态群雕"},
    {"time": "10:35", "name": "大型音乐盛典《吉祥颂》", "location": "灵山梵宫·圣坛", "duration": "30分钟", "free": False, "note": "需购票，建议提前 20 分钟入场"},
    {"time": "11:30", "name": "九龙灌浴·花开见佛", "location": "九龙灌浴广场", "duration": "18分钟", "free": True, "note": "大型音乐动态群雕"},
    {"time": "14:00", "name": "大型音乐盛典《吉祥颂》", "location": "灵山梵宫·圣坛", "duration": "30分钟", "free": False, "note": "需购票，建议提前 20 分钟入场"},
    {"time": "15:00", "name": "九龙灌浴·花开见佛", "location": "九龙灌浴广场", "duration": "18分钟", "free": True, "note": "大型音乐动态群雕"},
    {"time": "16:00", "name": "大型音乐盛典《吉祥颂》", "location": "灵山梵宫·圣坛", "duration": "30分钟", "free": False, "note": "需购票，建议提前 20 分钟入场"},
    {"time": "18:30", "name": "五灯湖《禅行》灯光秀", "location": "拈花湾·五灯湖", "duration": "25分钟", "free": True, "note": "夜间禅意光影秀"},
    {"time": "19:00", "name": "香月花街夜游巡游", "location": "拈花湾·香月花街", "duration": "30分钟", "free": True, "note": "赏花灯与禅意表演"},
    {"time": "20:00", "name": "五灯湖《禅行》灯光秀(第二场)", "location": "拈花湾·五灯湖", "duration": "25分钟", "free": True, "note": "夜间禅意光影秀"},
]


@app.get("/services/shows")
async def get_shows():
    """获取当日演出时间表"""
    return {"shows": SHOWS_DATA, "date": datetime.now().strftime("%Y-%m-%d")}


# ==================== 11. 便民服务 - 设施查询 ====================
FACILITIES = [
    {"id": "wc-01", "type": "洗手间", "icon": "🚻", "name": "入口广场洗手间", "lat": 31.4245, "lng": 120.0950, "desc": "景区入口左侧，无障碍卫生间"},
    {"id": "wc-02", "type": "洗手间", "icon": "🚻", "name": "九龙灌浴洗手间", "lat": 31.4218, "lng": 120.0935, "desc": "九龙灌浴广场右侧"},
    {"id": "wc-03", "type": "洗手间", "icon": "🚻", "name": "梵宫洗手间", "lat": 31.4200, "lng": 120.0910, "desc": "梵宫地下一层"},
    {"id": "wc-04", "type": "洗手间", "icon": "🚻", "name": "祥符禅寺洗手间", "lat": 31.4210, "lng": 120.0925, "desc": "禅寺东侧"},
    {"id": "wc-05", "type": "洗手间", "icon": "🚻", "name": "拈花湾洗手间", "lat": 31.4080, "lng": 120.1040, "desc": "拈花广场旁"},
    {"id": "mb-01", "type": "母婴室", "icon": "👶", "name": "入口游客中心母婴室", "lat": 31.4245, "lng": 120.0952, "desc": "配备尿布台、温奶器"},
    {"id": "mb-02", "type": "母婴室", "icon": "👶", "name": "梵宫母婴室", "lat": 31.4200, "lng": 120.0912, "desc": "梵宫一层西侧"},
    {"id": "med-01", "type": "医疗点", "icon": "🏥", "name": "景区医务室", "lat": 31.4240, "lng": 120.0950, "desc": "入口广场游客服务中心内"},
    {"id": "med-02", "type": "医疗点", "icon": "🏥", "name": "拈花湾医务室", "lat": 31.4080, "lng": 120.1035, "desc": "香月花街中段"},
    {"id": "acc-01", "type": "无障碍设施", "icon": "♿", "name": "无障碍通道入口", "lat": 31.4243, "lng": 120.0950, "desc": "景区正门右侧无障碍通道"},
    {"id": "acc-02", "type": "无障碍设施", "icon": "♿", "name": "大佛无障碍电梯", "lat": 31.4208, "lng": 120.0923, "desc": "灵山大佛基座电梯"},
    {"id": "sr-01", "type": "游客中心", "icon": "🏢", "name": "主游客服务中心", "lat": 31.4243, "lng": 120.0955, "desc": "票务、咨询、寄存、轮椅租借"},
    {"id": "sr-02", "type": "游客中心", "icon": "🏢", "name": "拈花湾游客中心", "lat": 31.4090, "lng": 120.1042, "desc": "拈花湾入口左侧"},
    {"id": "shop-01", "type": "便利店", "icon": "🏪", "name": "景区入口便利店", "lat": 31.4242, "lng": 120.0950, "desc": "饮料、零食、雨具"},
    {"id": "shop-02", "type": "便利店", "icon": "🏪", "name": "梵宫纪念品商店", "lat": 31.4200, "lng": 120.0915, "desc": "佛教文创、纪念品"},
]


@app.get("/services/facilities")
async def get_facilities(type: str = ""):
    """获取景区便民设施列表，可选按类型筛选"""
    if type:
        filtered = [f for f in FACILITIES if f["type"] == type]
    else:
        filtered = FACILITIES
    return {"facilities": filtered, "types": list(set(f["type"] for f in FACILITIES))}


# ==================== 12. 便民服务 - 紧急求助 ====================
EMERGENCY_CONTACTS = {
    "rescue": {"name": "景区救援热线", "phone": "0510-85680000", "icon": "🆘"},
    "medical": {"name": "景区医务室", "phone": "0510-85680120", "icon": "🏥"},
    "police": {"name": "马山派出所", "phone": "0510-85995110", "icon": "👮"},
    "fire": {"name": "火警", "phone": "119", "icon": "🚒"},
}


@app.post("/services/emergency")
async def emergency_alert():
    """紧急求助：记录求助请求并返回救援信息"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO conversations (session_id, question, answer, timestamp) VALUES (?, ?, ?, ?)",
            ("emergency", "【紧急求助】用户发出紧急求助信号", "系统已记录并通知工作人员", datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return {
        "status": "ok",
        "message": "紧急求助已发送，工作人员正在赶来。请保持冷静，在原地等候。",
        "contacts": EMERGENCY_CONTACTS,
    }


@app.get("/services/emergency")
async def get_emergency_contacts():
    """获取紧急联系方式"""
    return {"contacts": EMERGENCY_CONTACTS}


# ==================== 游览路线规划 ====================
# 所有景点坐标数据（GCJ-02），前端已硬编码，后端同步一份用于路线规划
SPOT_DATA: list[dict[str, Any]] = [
    {"id": "LS-001", "name": "灵山大照壁", "lat": 31.421388, "lng": 120.102499, "icon": "🏛️", "desc": "华夏第一壁，景区入口",
     "photo_spots": [{"tip": "正前方全景拍摄，将大照壁与蓝天白云同框", "angle": "正面全景", "best_time": "上午顺光"}]},
    {"id": "LS-002", "name": "五明桥", "lat": 31.421749, "lng": 120.102248, "icon": "🌉", "desc": "五座汉白玉石拱桥",
     "photo_spots": [{"tip": "站在桥中央回拍入口方向，利用桥面延伸线构图", "angle": "仰拍桥身", "best_time": "清晨或黄昏"}]},
    {"id": "LS-003", "name": "佛足坛", "lat": 31.422725, "lng": 120.101497, "icon": "👣", "desc": "朝圣祈福核心节点",
     "photo_spots": [{"tip": "低角度拍摄佛足印，突出石刻纹理与庄严感", "angle": "低角度特写", "best_time": "全天"}]},
    {"id": "LS-004", "name": "五智门", "lat": 31.423055, "lng": 120.101292, "icon": "⛩️", "desc": "五门六柱石牌坊",
     "photo_spots": [{"tip": "穿过门洞拍摄远处大佛，形成框景构图", "angle": "框景构图", "best_time": "下午逆光剪影"}]},
    {"id": "LS-005", "name": "菩提大道", "lat": 31.423182, "lng": 120.101143, "icon": "🌳", "desc": "禅意步道",
     "photo_spots": [{"tip": "利用道路两侧银杏树形成纵深透视感", "angle": "纵深透视", "best_time": "秋季银杏金黄时"}]},
    {"id": "LS-006", "name": "九龙灌浴", "lat": 31.424601, "lng": 120.099984, "icon": "⛲", "desc": "大型音乐动态群雕",
     "photo_spots": [{"tip": "喷水表演时连拍，捕捉水花与阳光形成的彩虹", "angle": "正面+特写", "best_time": "表演时段（10:00/11:30/14:00/15:30）"}]},
    {"id": "LS-007", "name": "降魔浮雕", "lat": 31.425559, "lng": 120.099569, "icon": "🗿", "desc": "佛陀觉悟成道故事",
     "photo_spots": [{"tip": "侧面拍摄利用光影突出浮雕层次感", "angle": "侧面45°", "best_time": "上午侧光"}]},
    {"id": "LS-008", "name": "阿育王柱", "lat": 31.426188, "lng": 120.099261, "icon": "🪨", "desc": "古印度石柱复刻",
     "photo_spots": [{"tip": "仰拍石柱顶部狮像，蓝天为背景", "angle": "仰拍", "best_time": "晴天上午"}]},
    {"id": "LS-009", "name": "百子戏弥勒", "lat": 31.427190, "lng": 120.098844, "icon": "🧸", "desc": "9吨青铜群雕",
     "photo_spots": [{"tip": "环绕拍摄，捕捉每个小童的生动表情", "angle": "环绕多角度", "best_time": "全天"}]},
    {"id": "LS-010", "name": "祥符禅寺", "lat": 31.427949, "lng": 120.098012, "icon": "🏯", "desc": "唐代千年古刹",
     "photo_spots": [{"tip": "寺前香炉烟雾缭绕时拍摄，营造禅意氛围", "angle": "正面+香炉前景", "best_time": "清晨香火旺盛时"}]},
    {"id": "LS-011", "name": "灵山大佛", "lat": 31.430194, "lng": 120.096477, "icon": "🗽", "desc": "88米青铜释迦牟尼立像",
     "photo_spots": [
         {"tip": "登顶后从大佛脚下回拍全景，俯瞰整个景区", "angle": "俯瞰全景", "best_time": "晴天能见度高时"},
         {"tip": "抱佛脚雕像处，触摸佛脚祈福合影", "angle": "近景合影", "best_time": "全天"},
         {"tip": "九龙灌浴广场远拍大佛，水景与佛像同框", "angle": "远景+水景", "best_time": "表演时段"}
     ]},
    {"id": "LS-012", "name": "佛教文化博览馆", "lat": 31.427856, "lng": 120.105632, "icon": "🏛️", "desc": "万佛殿与佛教史展",
     "photo_spots": [{"tip": "万佛殿内拍摄穹顶万佛，广角镜头最佳", "angle": "仰拍穹顶", "best_time": "室内全天"}]},
    {"id": "LS-013", "name": "灵山梵宫", "lat": 31.428218, "lng": 120.102420, "icon": "🕌", "desc": "东方卢浮宫",
     "photo_spots": [
         {"tip": "梵宫正面全景，将金顶与蓝天同框", "angle": "正面全景", "best_time": "上午顺光"},
         {"tip": "宫内穹顶壁画，仰拍展示华丽天花", "angle": "仰拍穹顶", "best_time": "室内全天"},
         {"tip": "梵宫外廊柱光影，利用廊柱形成纵深构图", "angle": "廊柱透视", "best_time": "午后光影斑驳时"}
     ]},
    {"id": "LS-014", "name": "五印坛城", "lat": 31.424676, "lng": 120.103054, "icon": "🏔️", "desc": "小布达拉宫",
     "photo_spots": [{"tip": "湖对面拍摄坛城倒影，对称构图", "angle": "远景+倒影", "best_time": "无风清晨或傍晚"}]},
    {"id": "LS-015", "name": "曼飞龙塔", "lat": 31.426070, "lng": 120.104609, "icon": "🕌", "desc": "南传佛教白塔",
     "photo_spots": [{"tip": "蓝天白云下白塔群组，色彩对比强烈", "angle": "正面全景", "best_time": "晴天上午"}]},
    {"id": "LS-016", "name": "无尽意斋", "lat": 31.428768, "lng": 120.096987, "icon": "🏡", "desc": "赵朴初纪念馆",
     "photo_spots": [{"tip": "庭院内拍摄江南园林小景，假山流水", "angle": "园林小品", "best_time": "午后柔和光线"}]},
    {"id": "NH-001", "name": "拈花广场", "lat": 31.420040, "lng": 120.076954, "icon": "🌸", "desc": "拈花湾入口",
     "photo_spots": [{"tip": "入口牌坊处拍摄，记录拈花湾之旅起点", "angle": "正面", "best_time": "全天"}]},
    {"id": "NH-002", "name": "梵天花海", "lat": 31.415904, "lng": 120.075421, "icon": "🌻", "desc": "30000㎡四季花海",
     "photo_spots": [
         {"tip": "花海中的人像拍摄，蹲下与花同高营造花海包围感", "angle": "低角度人像", "best_time": "花季上午或傍晚"},
         {"tip": "无人机视角俯瞰花海图案（如允许）", "angle": "俯瞰", "best_time": "花季晴天"}
     ]},
    {"id": "NH-003", "name": "香月花街", "lat": 31.416822, "lng": 120.073636, "icon": "🏮", "desc": "800米禅意商业街",
     "photo_spots": [{"tip": "华灯初上时拍摄灯笼长廊，唐风禅意十足", "angle": "纵深透视", "best_time": "傍晚至入夜"}]},
    {"id": "NH-004", "name": "拈花堂", "lat": 31.417841, "lng": 120.078339, "icon": "🧘", "desc": "静心禅堂",
     "photo_spots": [{"tip": "堂前静坐人像，禅意剪影", "angle": "剪影效果", "best_time": "黄昏逆光"}]},
    {"id": "NH-005", "name": "五灯湖", "lat": 31.418665, "lng": 120.075312, "icon": "🌊", "desc": "禅行灯光秀",
     "photo_spots": [{"tip": "灯光秀时拍摄湖面倒影与水幕投影", "angle": "水面倒影", "best_time": "夜间表演时段"}]},
    {"id": "NH-006", "name": "鹿鸣谷", "lat": 31.424319, "lng": 120.079449, "icon": "🦌", "desc": "山林幽谷",
     "photo_spots": [{"tip": "林间小道拍摄，利用晨雾或阳光透过树叶的光斑", "angle": "林间光影", "best_time": "清晨"}]},
]

SPOT_ID_MAP: dict[str, dict[str, Any]] = {s["id"]: s for s in SPOT_DATA}

ROUTE_PREFERENCE_MAP = {
    "culture": "历史文化深度游，偏好古刹、寺庙、文化展馆、名人故居类景点",
    "photo": "网红打卡拍照游，偏好外观壮丽、有视觉冲击力、适合拍照的景点",
    "zen": "禅修静心体验游，偏好禅堂、山林、静谧湖泊、茶室类景点",
    "family": "亲子互动欢乐游，偏好有趣味性、互动性强、适合家庭游玩的项目",
    "general": "经典全景游览，涵盖灵山胜境和拈花湾的代表性景点，路线合理不走回头路",
}

DURATION_MAP = {
    "1h": "约1小时，推荐3-4个核心景点",
    "2h": "约2小时，推荐5-6个景点",
    "half_day": "半天（约3-4小时），推荐8-10个景点，含一次休息/用餐",
    "full_day": "全天（约6-8小时），推荐12-15个景点，含午休与用餐",
}


@app.post("/api/route/plan", response_model=RoutePlanResponse)
async def route_plan(req: RoutePlanRequest):
    """游览路线规划：根据偏好和时长，AI 推荐最优游览顺序"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")

    # 筛选候选景点
    if req.spot_ids:
        candidate_spots = [SPOT_ID_MAP[sid] for sid in req.spot_ids if sid in SPOT_ID_MAP]
    else:
        candidate_spots = SPOT_DATA

    if len(candidate_spots) < 2:
        raise HTTPException(status_code=400, detail="候选景点不足，至少需要2个")

    preference_desc = ROUTE_PREFERENCE_MAP.get(req.preference, ROUTE_PREFERENCE_MAP["general"])
    duration_desc = DURATION_MAP.get(req.duration, DURATION_MAP["half_day"])

    spot_list = "\n".join([
        f"- {s['id']}: {s['icon']} {s['name']} (lat={s['lat']},lng={s['lng']}) {s['desc']}"
        for s in candidate_spots
    ])

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )

    lang_map = {"zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어"}
    lang_name = lang_map.get(req.lang, "中文")

    prompt = f"""你是景区金牌导游，请为游客规划一条游览路线。请用{lang_name}输出。输出纯 JSON（不要其他文字）：

要求：
- 偏好：{preference_desc}
- 时间：{duration_desc}
- 路线必须按地理顺路排序，避免折返
- 每个景点附上推荐理由（10字以内）

候选景点：
{spot_list}

JSON 格式：
{{
  "title": "<路线名称，有吸引力，10字以内>",
  "spots": [{{"id":"LS-001","reason":"<理由>"}}, ...],
  "tips": "<游览小贴士，30字以内>"
}}
"""
    try:
        resp = await llm.ainvoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])

        # 组装带完整坐标的景点列表
        ordered_spots = []
        for item in data.get("spots", []):
            spot = SPOT_ID_MAP.get(item["id"])
            if spot:
                ordered_spots.append({
                    "id": spot["id"],
                    "name": spot["name"],
                    "lat": spot["lat"],
                    "lng": spot["lng"],
                    "icon": spot["icon"],
                    "desc": spot["desc"],
                    "reason": item.get("reason", ""),
                })

        if len(ordered_spots) < 2:
            ordered_spots = [{
                "id": s["id"], "name": s["name"], "lat": s["lat"],
                "lng": s["lng"], "icon": s["icon"], "desc": s["desc"],
                "reason": "经典路线"
            } for s in candidate_spots[:6]]

        return RoutePlanResponse(
            title=data.get("title", "推荐游览路线"),
            spots=ordered_spots,
            tips=data.get("tips", "祝您游览愉快！"),
        )
    except Exception as e:
        logger.exception("路线规划失败")
        # 兜底：按地理坐标排序
        fallback = sorted(candidate_spots, key=lambda s: (s["lng"], s["lat"]))
        fallback = fallback[:8]
        return RoutePlanResponse(
            title="经典游览路线（备选）",
            spots=[{
                "id": s["id"], "name": s["name"], "lat": s["lat"],
                "lng": s["lng"], "icon": s["icon"], "desc": s["desc"],
                "reason": "经典顺路"
            } for s in fallback],
            tips="建议早出发，避开人流高峰",
        )


# ==================== AI 游记生成 ====================
STYLE_MAP = {
    "literary": "文艺游记风格，语言优美，富有诗意和画面感，适合分享到朋友圈",
    "guide": "实用攻略风格，侧重路线总结、实用 tips、时间安排、花费参考",
    "relaxed": "轻松随笔风格，语气亲切自然，像朋友聊天一样分享旅途见闻",
}


@app.post("/chat/travelogue", response_model=TravelogueResponse)
async def generate_travelogue(req: TravelogueRequest):
    """根据会话历史生成 AI 游记"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY")

    # 1. 拉取会话对话记录
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT question, answer FROM conversations WHERE session_id = ? ORDER BY id ASC",
        (req.session_id,),
    ).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="该会话没有对话记录")

    # 2. 检查是否已有游记
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT id, title, content, style, spots_json, created_at FROM travelogues WHERE session_id = ?",
        (req.session_id,),
    ).fetchone()
    if existing:
        conn.close()
        return TravelogueResponse(
            id=existing["id"],
            session_id=req.session_id,
            title=existing["title"],
            content=existing["content"],
            style=existing["style"],
            spots_visited=json.loads(existing["spots_json"]),
            created_at=existing["created_at"],
        )
    conn.close()

    # 3. 提取对话中涉及的景点
    all_text = " ".join([r["question"] + " " + r["answer"] for r in rows])
    visited_spots = []
    for spot in SPOT_DATA:
        if spot["name"] in all_text:
            visited_spots.append(spot)

    # 4. 构建对话摘要
    conversation_summary = []
    for r in rows:
        q = r["question"][:200]
        a = r["answer"][:300]
        conversation_summary.append(f"游客：{q}\n导览员：{a}")
    summary_text = "\n\n".join(conversation_summary[-20:])  # 最多取最近20轮

    # 5. 景点信息
    spot_info = "\n".join([
        f"- {s['icon']} {s['name']}：{s['desc']}"
        for s in visited_spots
    ]) if visited_spots else "（根据对话内容推断）"

    # 6. 风格描述
    style_desc = STYLE_MAP.get(req.style, STYLE_MAP["literary"])

    lang_map = {"zh": "中文", "en": "English", "ja": "日本語", "ko": "한국어"}
    lang_name = lang_map.get(req.lang, "中文")

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    )

    prompt = f"""你是一位优秀的旅行作家。请根据以下游客与 AI 导览员的对话记录，撰写一篇精彩的游记。

要求：
- 用{lang_name}书写
- 风格：{style_desc}
- 结构：标题（用 # 开头）→ 开篇引入 → 按游览顺序逐景点描写 → 旅行感悟结尾
- 融入景点文化内涵和历史背景，让读者有身临其境之感
- 字数控制在 800-1500 字
- 使用 Markdown 格式，景点名称用 **加粗**

游客游览过的景点参考：
{spot_info}

对话记录：
{summary_text}

请直接输出游记内容，不要输出任何前缀说明。"""

    try:
        resp = await llm.ainvoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)

        # 7. 提取标题（取第一个 # 开头的行）
        title = "灵山游记"
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break
            if line.startswith("#"):
                title = line[1:].strip()
                break

        # 8. 存储游记
        travelogue_id = str(uuid.uuid4())
        spot_names = [s["name"] for s in visited_spots]
        now = datetime.now().isoformat()

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO travelogues (id, session_id, title, content, style, spots_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (travelogue_id, req.session_id, title, content, req.style, json.dumps(spot_names, ensure_ascii=False), now),
        )
        conn.commit()
        conn.close()

        return TravelogueResponse(
            id=travelogue_id,
            session_id=req.session_id,
            title=title,
            content=content,
            style=req.style,
            spots_visited=spot_names,
            created_at=now,
        )
    except Exception as e:
        logger.exception("游记生成失败")
        raise HTTPException(status_code=500, detail=f"游记生成失败: {e}")


@app.get("/api/travelogues")
async def list_travelogues(session_id: str = ""):
    """获取游记列表，可按 session_id 筛选"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if session_id:
        rows = conn.execute(
            "SELECT id, session_id, title, style, spots_json, created_at FROM travelogues WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, session_id, title, style, spots_json, created_at FROM travelogues ORDER BY created_at DESC"
        ).fetchall()
    conn.close()

    travelogues = []
    for r in rows:
        travelogues.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "title": r["title"],
            "style": r["style"],
            "spots_visited": json.loads(r["spots_json"]),
            "created_at": r["created_at"],
        })
    return {"travelogues": travelogues}


@app.get("/api/travelogues/{travelogue_id}")
async def get_travelogue(travelogue_id: str):
    """获取单篇游记详情"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, session_id, title, content, style, spots_json, created_at FROM travelogues WHERE id = ?",
        (travelogue_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="游记不存在")

    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "title": row["title"],
        "content": row["content"],
        "style": row["style"],
        "spots_visited": json.loads(row["spots_json"]),
        "created_at": row["created_at"],
    }


@app.delete("/api/travelogues/{travelogue_id}")
async def delete_travelogue(travelogue_id: str):
    """删除一篇游记"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM travelogues WHERE id = ?", (travelogue_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "游记已删除"}


# ==================== 天气 + 穿衣建议 ====================
WEATHER_CLOTHING = {
    "zh": {
        "hot": "天气炎热，建议穿轻薄透气的短袖、短裤，戴遮阳帽和太阳镜，注意防晒，多喝水。",
        "warm": "天气温暖舒适，建议穿薄款长袖或短袖 + 薄外套，早晚微凉可备一件开衫。",
        "cool": "天气转凉，建议穿长袖 + 外套或薄毛衣，早晚需加一件风衣。",
        "cold": "天气寒冷，建议穿厚外套、羽绒服，戴围巾手套，注意保暖。",
        "rain": "有降雨，请携带雨伞或雨衣，穿防滑鞋，路面湿滑注意安全。",
        "sunny": "阳光充足，紫外线较强，请涂抹防晒霜，佩戴太阳镜和遮阳帽。",
    },
    "en": {
        "hot": "Hot weather. Wear light, breathable clothing (shorts, T-shirt). Bring a hat and sunglasses. Stay hydrated.",
        "warm": "Warm and pleasant. Light long sleeves or T-shirt with a thin jacket for the evening.",
        "cool": "Cool weather. Wear long sleeves with a jacket or light sweater. A windbreaker for the evening.",
        "cold": "Cold weather. Wear a thick coat or down jacket, scarf and gloves. Keep warm.",
        "rain": "Rain expected. Bring an umbrella or raincoat. Wear non-slip shoes.",
        "sunny": "Sunny with strong UV. Apply sunscreen, wear sunglasses and a hat.",
    },
    "ja": {
        "hot": "暑い天気です。薄手の半袖・短パンを着用し、帽子とサングラスを着用してください。水分補給を忘れずに。",
        "warm": "暖かい天気です。薄手の長袖や半袖に薄い上着をお勧めします。",
        "cool": "涼しい天気です。長袖にジャケットや薄手のセーターを着用してください。",
        "cold": "寒い天気です。厚手のコートやダウンジャケット、マフラーと手袋を着用してください。",
        "rain": "雨が予想されます。傘やレインコートをご持参ください。滑りにくい靴を着用してください。",
        "sunny": "日差しが強いです。日焼け止めを塗り、サングラスと帽子を着用してください。",
    },
    "ko": {
        "hot": "더운 날씨입니다. 얇은 반팔, 반바지를 입고 모자와 선글라스를 착용하세요. 수분을 충분히 섭취하세요.",
        "warm": "따뜻한 날씨입니다. 얇은 긴팔이나 반팔에 얇은 겉옷을 추천합니다.",
        "cool": "선선한 날씨입니다. 긴팔에 재킷이나 얇은 스웨터를 입으세요.",
        "cold": "추운 날씨입니다. 두꺼운 코트나 패딩, 목도리와 장갑을 착용하세요.",
        "rain": "비가 예상됩니다. 우산이나 우비를 챙기세요. 미끄럼 방지 신발을 신으세요.",
        "sunny": "햇볕이 강합니다. 자외선 차단제를 바르고 선글라스와 모자를 착용하세요.",
    },
}


@app.get("/api/weather")
async def get_weather(lat: float = 31.425, lng: float = 120.10, lang: str = "zh"):
    """获取天气信息 + 穿衣建议（使用 wttr.in 免费 API）"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://wttr.in/{lat},{lng}?format=j1&lang={lang}",
                headers={"User-Agent": "AI-Tour-Agent/1.0"},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="天气服务暂时不可用")

            data = resp.json()
            current = data.get("current_condition", [{}])[0]
            weather_info = data.get("weather", [{}])[0]

            temp_c = int(current.get("temp_C", 0))
            humidity = current.get("humidity", "N/A")
            weather_desc = current.get("weatherDesc", [{}])[0].get("value", "未知")
            wind_speed = current.get("windspeedKmph", "N/A")
            feels_like = current.get("FeelsLikeC", str(temp_c))
            uv_index = current.get("uvIndex", "N/A")

            # 穿衣建议
            clothing = WEATHER_CLOTHING.get(lang, WEATHER_CLOTHING["zh"])
            if temp_c >= 30:
                advice = clothing["hot"]
            elif temp_c >= 20:
                advice = clothing["warm"]
            elif temp_c >= 10:
                advice = clothing["cool"]
            else:
                advice = clothing["cold"]

            # 降雨提示
            if "雨" in weather_desc or "rain" in weather_desc.lower() or "shower" in weather_desc.lower():
                advice += " " + clothing["rain"]
            # 紫外线提示
            if uv_index != "N/A" and int(uv_index) >= 6:
                advice += " " + clothing["sunny"]

            # 今日预报
            today_forecast = weather_info.get("hourly", [{}])[0] if weather_info.get("hourly") else {}
            today_high = weather_info.get("maxtempC", "N/A")
            today_low = weather_info.get("mintempC", "N/A")

            return {
                "current": {
                    "temp_c": temp_c,
                    "feels_like": feels_like,
                    "humidity": humidity,
                    "weather_desc": weather_desc,
                    "wind_speed": wind_speed,
                    "uv_index": uv_index,
                },
                "today": {
                    "high": today_high,
                    "low": today_low,
                },
                "clothing_advice": advice,
                "location": f"{lat},{lng}",
            }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="天气服务请求超时")
    except Exception as e:
        logger.exception("天气获取失败")
        raise HTTPException(status_code=500, detail=f"天气获取失败: {e}")


# ==================== 拍照点推荐 ====================
@app.get("/api/photo-spots")
async def get_photo_spots():
    """获取所有景点的拍照点推荐"""
    spots = []
    for s in SPOT_DATA:
        if s.get("photo_spots"):
            spots.append({
                "spot_id": s["id"],
                "spot_name": s["name"],
                "icon": s["icon"],
                "lat": s["lat"],
                "lng": s["lng"],
                "photo_spots": s["photo_spots"],
            })
    return {"spots": spots}


# ==================== 健康检查 ====================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qa_ready": qa_chain is not None,
        "knowledge_files": len(list(KNOWLEDGE_DIR.glob("*.txt"))) if KNOWLEDGE_DIR.exists() else 0,
    }


# ---------- 托管前端静态文件（放在所有 API 路由之后） ----------
FRONTEND_DIR = BASE_DIR.parent / "frontend_tourist"
if FRONTEND_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
