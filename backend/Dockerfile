# AI 数字人景区导览服务 - 部署镜像
# 构建: docker build -t tour-agent .
# 运行: docker run -p 8000:8000 -e DEEPSEEK_API_KEY=sk-xxx -e VOLCANO_API_KEY=yyy tour-agent

FROM python:3.10-slim

WORKDIR /app

# 系统依赖（Edge TTS 播放需要 ffmpeg，可选）
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码与示例知识库
COPY . .

# 预创建目录
RUN mkdir -p knowledge_base chroma_db

EXPOSE 8000

# 环境变量 DEEPSEEK_API_KEY / VOLCANO_API_KEY 通过 docker run -e 传入
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
