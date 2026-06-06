# AI 数字人景区导览服务（灵山胜境）

> 比赛项目：Python FastAPI + LangChain RAG + Chroma + DeepSeek API  
> 示范景区：**江苏无锡灵山胜境 + 拈花湾**  
> 游客端（语音/文本/SSE 流式 + Canvas 数字人动画）+ 管理后台（知识库 + 数据大屏 + 形象配置）

## 项目结构

```
AI_Tour_Agent/
├── backend/
│   ├── app.py              # FastAPI 主服务（含个性化推荐/文档管理/兴趣标签接口）
│   ├── rag_chain.py        # RAG 向量库构建与 QA 链（含 5 种兴趣模式）
│   ├── text_utils.py       # Markdown→纯文本→TTS 文本转换
│   ├── requirements.txt
│   ├── Dockerfile          # Railway 部署
│   ├── knowledge_base/     # 知识库文档（.txt）
│   └── chroma_db/          # Chroma 向量库（运行 rag_chain.py 后生成）
├── frontend_admin/
│   └── index.html          # 管理后台（上传+大屏+情感+形象配置+文档管理）
├── frontend_tourist/
│   └── index.html          # 游客端（语音/文本+Canvas 数字人动画+兴趣标签）
├── docs/
│   ├── 产品设计文档.md
│   └── 演示视频脚本.md
└── README.md
```

## 快速开始

### 1. 环境准备

```powershell
cd backend
python -m venv venv

# Windows PowerShell（注意是 Activate.ps1，不是 activate）
.\venv\Scripts\Activate.ps1

# 若提示“无法加载，因为在此系统上禁止运行脚本”，先执行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Windows CMD
# venv\Scripts\activate.bat

pip install -r requirements.txt
```

> **说明**：`requirements.txt` 请勿写中文注释，否则 Windows 下 pip 可能因 GBK 编码报错。

### 2. 配置 API Key

```powershell
$env:DEEPSEEK_API_KEY="你的DeepSeek密钥"
```

或复制 `backend/.env.example` 为 `backend/.env` 并填写。

**重要**：DeepSeek **不提供** Embeddings 接口（`text-embedding-ada-002` 会返回 404）。  
本项目默认用 **本地中文向量模型** `BAAI/bge-small-zh-v1.5` 建库，DeepSeek 仅用于 **对话生成**。  
首次运行 `rag_chain.py` 会自动下载约 100MB 模型，请保持网络畅通。

若你已有 `AI-Agent-Dev/local_models/m3e-base`，可在 `.env` 中设置：
`LOCAL_EMBEDDING_MODEL=E:/ai_study/AI-Agent-Dev/local_models/m3e-base`

### 3. 导入示范景区知识库（首次或更新资料后）

将桌面「示范景区公开资料包」转为 `knowledge_base/*.txt`：

```powershell
cd backend
python scripts/import_lingshan_data.py
```

知识库包含：游览指南、景点结构化数据集、游客行为分析摘要（777 条灵山相关记录）。

### 4. 构建向量库（必须先执行）

```bash
cd backend
python rag_chain.py
```

成功后会看到 `chroma_db/` 目录生成，并输出一次测试问答。

### 5. 启动后端

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

访问 API 文档：http://127.0.0.1:8000/docs

### 6. 打开前端

用浏览器直接打开（或用 Live Server）：

- 游客端：`frontend_tourist/index.html`
- 管理后台：`frontend_admin/index.html`

页面顶部可修改 **后端 API 地址**（默认 `http://127.0.0.1:8000`）。

## 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat/text` | 文本问答（带 session 记忆） |
| POST | `/chat/text/stream` | 文本问答 SSE 流式（打字机+数字人表情指令） |
| GET | `/admin/interests` | 获取兴趣标签列表（5 种个性化模式） |
| POST | `/admin/upload` | 上传知识库文件 |
| GET | `/admin/documents` | 知识库文件列表 |
| DELETE | `/admin/documents/{filename}` | 删除知识库文件并重建 |
| GET | `/admin/stats` | 数据大屏统计 |
| POST | `/voice-to-text` | 语音识别 |
| POST | `/text-to-speech` | 语音合成（支持 voice 参数切换角色） |
| POST | `/admin/sentiment` | 情感分析 |

## Docker 部署（Railway）

```bash
cd backend
docker build -t tour-agent .
docker run -p 8000:8000 -e DEEPSEEK_API_KEY=sk-xxx tour-agent
```

部署后需在容器内执行一次 `python rag_chain.py`，或通过管理后台上传文档触发重建。

## 技术栈

- **后端**：FastAPI、LangChain、Chroma、DeepSeek API
- **语音**：faster-whisper（ASR）、edge-tts（TTS，多角色）
- **前端**：原生 HTML/JS、ECharts、Canvas 2D 数字人动画
- **数字人**：Canvas 自绘（idle/thinking/speaking 三表情 + 眨眼 + 口型同步）

## 注意事项

1. `faster-whisper` 首次运行会自动下载 `base` 模型，需联网。
2. Embeddings 使用 DeepSeek OpenAI 兼容接口，请确认账号支持 `text-embedding-ada-002`。
3. 前端通过 `file://` 打开时，麦克风可能受限，建议用本地 HTTP 服务器打开游客端。
