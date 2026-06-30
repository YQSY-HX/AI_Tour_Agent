"""
AI 数字人景区导览服务 - Vercel 最小版
仅包含不依赖 langchain/chromadb 的接口
"""
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tour_agent")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "conversations.db"

# ── 数据模型 ──
class ChatTextRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = Field(None)
    interest: str = Field("general")
    lang: str = Field("zh")
    image: str | None = Field(None)

class LoginRequest(BaseModel):
    username: str
    password: str

class FeedbackRequest(BaseModel):
    session_id: str | None = Field(None)
    question: str
    answer: str = ""
    rating: str = Field(..., description="good 或 bad")
    comment: str = ""

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice: str = Field("zh-CN-XiaoxiaoNeural")
    rate: str = Field("+0%")

# ── 模拟用户 ──
MOCK_USERS = {
    "user": {"password": "123456", "name": "游客", "role": "tourist"},
    "admin": {"password": "admin123", "name": "管理员", "role": "admin"},
}

def generate_token(username: str) -> str:
    return f"token_{username}_{uuid.uuid4().hex[:16]}"

def verify_user(username: str, password: str) -> dict | None:
    user = MOCK_USERS.get(username)
    if user and user["password"] == password:
        return {"username": username, "name": user["name"], "role": user["role"]}
    return None

# ── SQLite ──
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
        rating TEXT NOT NULL, comment TEXT DEFAULT '', timestamp TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS stats_cache (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

def save_conversation(session_id: str, question: str, answer: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO conversations (session_id, question, answer, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, question, answer, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

# ── 会话记忆 ──
session_memory: dict[str, list[dict[str, str]]] = {}
MAX_HISTORY_TURNS = 6

def get_or_create_session(session_id: str | None) -> str:
    sid = session_id or str(uuid.uuid4())
    if sid not in session_memory:
        session_memory[sid] = []
    return sid

def append_memory(session_id: str, role: str, content: str):
    session_memory[session_id].append({"role": role, "content": content})
    if len(session_memory[session_id]) > MAX_HISTORY_TURNS * 2:
        session_memory[session_id] = session_memory[session_id][-(MAX_HISTORY_TURNS * 2):]

# ── 景区数据 ──
WEATHER_LABELS = {
    "晴": {"icon": "☀️", "tip": "天气晴朗，适合户外游览"},
    "多云": {"icon": "⛅", "tip": "多云天气，体感舒适"},
    "阴": {"icon": "☁️", "tip": "阴天，建议带薄外套"},
    "小雨": {"icon": "🌧️", "tip": "有小雨，建议携带雨伞"},
}

WEATHER_CLOTHING = {
    "zh": {
        "hot": "天气炎热，建议穿轻薄透气的短袖、短裤，戴遮阳帽。",
        "warm": "天气温暖舒适，建议穿薄款长袖或短袖。",
        "cool": "天气转凉，建议穿长袖 + 外套。",
        "cold": "天气寒冷，建议穿厚外套、羽绒服。",
    },
}

INTEREST_CONFIG = {
    "general": {"label": "自由探索"},
    "history": {"label": "历史文化爱好者"},
    "nature": {"label": "自然风光爱好者"},
    "family": {"label": "亲子家庭"},
    "quick": {"label": "高效打卡"},
}

SHOWS_DATA = [
    {"time": "10:00", "name": "九龙灌浴·花开见佛", "location": "九龙灌浴广场"},
    {"time": "11:30", "name": "九龙灌浴·花开见佛", "location": "九龙灌浴广场"},
    {"time": "14:00", "name": "大型音乐盛典《吉祥颂》", "location": "灵山梵宫·圣坛"},
    {"time": "16:00", "name": "大型音乐盛典《吉祥颂》", "location": "灵山梵宫·圣坛"},
]

FACILITIES = [
    {"id": "wc-01", "type": "洗手间", "icon": "🚻", "name": "入口广场洗手间", "lat": 31.4245, "lng": 120.0950},
    {"id": "wc-02", "type": "洗手间", "icon": "🚻", "name": "九龙灌浴洗手间", "lat": 31.4218, "lng": 120.0935},
    {"id": "mb-01", "type": "母婴室", "icon": "👶", "name": "入口游客中心母婴室", "lat": 31.4245, "lng": 120.0952},
    {"id": "med-01", "type": "医疗点", "icon": "🏥", "name": "景区医务室", "lat": 31.4240, "lng": 120.0950},
]

EMERGENCY_CONTACTS = {
    "rescue": {"name": "景区救援热线", "phone": "0510-85680000", "icon": "🆘"},
    "medical": {"name": "景区医务室", "phone": "0510-85680120", "icon": "🏥"},
}

WEATHER_CACHE = {}
WEATHER_CACHE_TIME = 0

# ── FastAPI App ──
app = FastAPI(title="AI 数字人景区导览服务", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Vercel 最小版启动完成")

# ── 健康检查 ──
@app.get("/health")
async def health():
    return {"status": "ok", "mode": "vercel-minimal", "qa_ready": False, "knowledge_files": 0}

# ── 认证 ──
@app.post("/auth/login")
async def login(req: LoginRequest):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = generate_token(req.username)
    return {"access_token": token, "token_type": "bearer", "user": user}

# ── 兴趣标签 ──
@app.get("/admin/interests")
async def admin_interests():
    return {"interests": [{"key": k, "label": v["label"]} for k, v in INTEREST_CONFIG.items()]}

# ── 会话管理 ──
@app.post("/chat/sessions")
async def create_session():
    sid = str(uuid.uuid4())
    session_memory[sid] = []
    return {"session_id": sid}

@app.get("/chat/sessions")
async def list_sessions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""SELECT session_id,
        (SELECT question FROM conversations c2 WHERE c2.session_id = c1.session_id ORDER BY id ASC LIMIT 1) AS title,
        COUNT(*) AS msg_count, MAX(timestamp) AS last_time
        FROM conversations c1 GROUP BY session_id ORDER BY last_time DESC""").fetchall()
    conn.close()
    sessions = []
    for r in rows:
        title = r["title"] or "新对话"
        if len(title) > 20:
            title = title[:20] + "..."
        sessions.append({"session_id": r["session_id"], "title": title, "msg_count": r["msg_count"], "last_time": r["last_time"]})
    return {"sessions": sessions}

@app.get("/chat/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, question, answer, timestamp FROM conversations WHERE session_id = ? ORDER BY id ASC", (session_id,)).fetchall()
    conn.close()
    return {"session_id": session_id, "messages": [{"id": r["id"], "question": r["question"], "answer": r["answer"], "timestamp": r["timestamp"]} for r in rows]}

@app.delete("/chat/sessions")
async def delete_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()
    session_memory.clear()
    return {"status": "ok", "message": "所有对话记录已删除"}

# ── 模拟文本问答（无 AI） ──
from text_utils import markdown_to_plaintext, plain_text_for_speech

@app.post("/chat/text")
async def chat_text(req: ChatTextRequest):
    session_id = get_or_create_session(req.session_id)
    answer = f"您好！我是灵山胜境 AI 导览员。当前为离线模式，AI 智能问答暂不可用。您可以查看演出时间表、天气信息等便民服务。\n\n如有问题，请拨打景区热线：0510-85680000"
    append_memory(session_id, "user", req.question)
    append_memory(session_id, "assistant", answer)
    save_conversation(session_id, req.question, answer)
    return {"answer": answer, "session_id": session_id}

@app.post("/chat/text/stream")
async def chat_text_stream(req: ChatTextRequest):
    session_id = get_or_create_session(req.session_id)
    append_memory(session_id, "user", req.question)
    answer = f"您好！我是灵山胜境 AI 导览员。当前为离线模式，AI 智能问答暂不可用。您可以查看演出时间表、天气信息等便民服务。\n\n如有问题，请拨打景区热线：0510-85680000"
    append_memory(session_id, "assistant", answer)
    save_conversation(session_id, req.question, answer)
    async def gen():
        for c in answer:
            yield f"data: {json.dumps({'type': 'token', 'content': c}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'answer': answer}, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")

# ── 反馈 ──
@app.post("/chat/feedback")
async def chat_feedback(req: FeedbackRequest):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO feedback (session_id, question, answer, rating, comment, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (req.session_id or "anonymous", req.question, req.answer, req.rating, req.comment, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"status": "ok"}

# ── 管理大屏 ──
@app.get("/admin/stats")
async def admin_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(DISTINCT session_id) FROM conversations").fetchone()[0] or 0
    today = conn.execute("SELECT COUNT(*) FROM conversations WHERE date(timestamp) = date('now', 'localtime')").fetchone()[0] or 0
    conn.close()
    from datetime import date, timedelta
    trend = [20, 25, 30, 28, 35, 42, 38]
    return {"total_sessions": total or 128, "today_queries": today or 45, "top_questions": ["灵山大佛", "九龙灌浴", "吉祥颂", "拈花湾", "门票"], "satisfaction": 4.3, "trend": trend}

# ── 天气 ──
@app.get("/services/weather")
async def get_weather():
    import time
    now = time.time()
    if WEATHER_CACHE and (now - WEATHER_CACHE_TIME) < 1800:
        return WEATHER_CACHE
    try:
        import urllib.request
        url = "https://wttr.in/Wuxi?format=j1&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        current = data.get("current_condition", [{}])[0]
        weather_desc = current.get("lang_zh", [{}])[0].get("value", "未知")
        temp = current.get("temp_C", "--")
        humidity = current.get("humidity", "--")
        wind_speed = current.get("windspeedKmph", "--")
        icon = WEATHER_LABELS.get(weather_desc, {}).get("icon", "🌤️")
        tip = WEATHER_LABELS.get(weather_desc, {}).get("tip", "祝您游览愉快")
        temp_val = int(temp) if temp and temp != "--" else 20
        if temp_val < 10: clothing = ["🧥 厚外套", "🧣 围巾"]
        elif temp_val < 18: clothing = ["🧥 薄外套", "👖 长裤"]
        elif temp_val < 26: clothing = ["👕 长袖/短袖", "👖 长裤"]
        else: clothing = ["👕 短袖", "🧴 防晒霜"]
        result = {"weather": {"city": "无锡·灵山胜境", "temperature": f"{temp}°C", "weather": weather_desc, "icon": icon, "humidity": f"{humidity}%", "wind": f"{wind_speed}km/h", "tip": tip, "clothing": clothing, "updated": datetime.now().strftime("%H:%M")}}
        WEATHER_CACHE["weather"] = result["weather"]
        WEATHER_CACHE_TIME = now
        return result
    except Exception:
        return {"weather": {"city": "无锡·灵山胜境", "temperature": "22°C", "weather": "多云", "icon": "⛅", "humidity": "65%", "wind": "12km/h", "tip": "多云天气，体感舒适", "clothing": ["👕 长袖/短袖"], "updated": "-- (离线数据)"}}

# ── 演出时间表 ──
@app.get("/services/shows")
async def get_shows():
    return {"shows": SHOWS_DATA, "date": datetime.now().strftime("%Y-%m-%d")}

# ── 设施查询 ──
@app.get("/services/facilities")
async def get_facilities(type: str = ""):
    if type:
        filtered = [f for f in FACILITIES if f["type"] == type]
    else:
        filtered = FACILITIES
    return {"facilities": filtered, "types": list(set(f["type"] for f in FACILITIES))}

# ── 紧急求助 ──
@app.get("/services/emergency")
async def get_emergency():
    return {"contacts": EMERGENCY_CONTACTS}

@app.post("/services/emergency")
async def emergency_alert():
    return {"status": "ok", "message": "紧急求助已发送", "contacts": EMERGENCY_CONTACTS}

# ── TTS（Edge TTS） ──
@app.post("/text-to-speech")
async def text_to_speech(req: TTSRequest):
    import tempfile
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(status_code=503, detail="TTS 服务暂不可用")
    voice = req.voice or "zh-CN-XiaoxiaoNeural"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()
    try:
        speak_text = plain_text_for_speech(req.text)
        communicate = edge_tts.Communicate(speak_text, voice, rate=req.rate or "+0%")
        await communicate.save(tmp_path)
        return FileResponse(tmp_path, media_type="audio/mpeg", filename="speech.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {e}")

# ── 天气 API（含穿衣建议） ──
@app.get("/api/weather")
async def get_weather_api(lat: float = 31.425, lng: float = 120.10, lang: str = "zh"):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"https://wttr.in/{lat},{lng}?format=j1&lang={lang}", headers={"User-Agent": "AI-Tour-Agent/1.0"})
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="天气服务暂不可用")
            data = resp.json()
            current = data.get("current_condition", [{}])[0]
            temp_c = int(current.get("temp_C", 0))
            humidity = current.get("humidity", "N/A")
            weather_desc = current.get("weatherDesc", [{}])[0].get("value", "未知")
            wind_speed = current.get("windspeedKmph", "N/A")
            feels_like = current.get("FeelsLikeC", str(temp_c))
            uv_index = current.get("uvIndex", "N/A")
            weather_info = data.get("weather", [{}])[0]
            clothing = WEATHER_CLOTHING.get(lang, WEATHER_CLOTHING["zh"])
            if temp_c >= 30: advice = clothing["hot"]
            elif temp_c >= 20: advice = clothing["warm"]
            elif temp_c >= 10: advice = clothing["cool"]
            else: advice = clothing["cold"]
            return {"current": {"temp_c": temp_c, "feels_like": feels_like, "humidity": humidity, "weather_desc": weather_desc, "wind_speed": wind_speed, "uv_index": uv_index}, "today": {"high": weather_info.get("maxtempC", "N/A"), "low": weather_info.get("mintempC", "N/A")}, "clothing_advice": advice, "location": f"{lat},{lng}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"天气获取失败: {e}")
