# AI 数字人景区导览服务 - Railway 部署镜像
# 构建: docker build -t tour-agent .
# 运行: docker run -p 8000:8000 -e DEEPSEEK_API_KEY=sk-xxx tour-agent

FROM python:3.10-slim

WORKDIR /app

# 系统依赖（faster-whisper 可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码与示例知识库
COPY . .

# 预创建目录
RUN mkdir -p knowledge_base chroma_db

# 构建时可选：若有知识库则建向量库（需构建参数传入 API Key，比赛环境建议在启动脚本中处理）
# 默认在首次部署后通过管理后台上传或手动 exec 运行 python rag_chain.py

EXPOSE 8000

# 环境变量 DEEPSEEK_API_KEY 通过 docker run -e 或 Railway Variables 传入
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
