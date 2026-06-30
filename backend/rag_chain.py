"""
RAG 向量库构建与问答链模块

功能：
  1. 从 knowledge_base/ 加载 .txt 文档
  2. 文本分割并向量化（默认火山 Embedding API，也支持 local/openai）
  3. 持久化到 chroma_db/
  4. 构建 RetrievalQA 问答链（DeepSeek Chat）

环境变量：
  DEEPSEEK_API_KEY       - 对话模型必填
  EMBEDDING_PROVIDER     - volcano（默认）| local | openai
  VOLCANO_API_KEY        - 火山引擎 API Key（Embedding 必填）
  VOLCANO_BASE_URL       - 火山引擎 Embedding 接口地址
  VOLCANO_EMBEDDING_MODEL - 火山 Embedding 模型名，默认 doubao-embedding-large
  LOCAL_EMBEDDING_MODEL  - 本地模型名（仅 provider=local 时使用）

说明：DeepSeek 不提供 Embeddings 接口，改用火山引擎 Embedding API。
"""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import load_dotenv
import httpx
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 约束模型输出为纯文本，减少 Markdown
QA_PROMPT = PromptTemplate(
    template="""你是「灵山胜境」AI 数字人导览员，服务江苏无锡灵山胜境及拈花湾景区。
资料涵盖：景点介绍、开放时间、门票、游览路线、历史文化、表演时刻、游客行为数据等。
{interest_instruction}
【回答要求】
1. 使用通俗易懂的中文，面向普通游客
2. 不要使用 Markdown（禁止 #、*、**、- 列表等符号）
3. 需要列举时，用「1. 2. 3.」数字编号，每项单独一行
4. 优先依据参考资料回答；资料未提及则说明「暂无相关信息」，不要编造
5. 涉及九龙灌浴、《吉祥颂》等表演时间，若资料有具体场次则如实引用
6. 当游客询问景点位置、路线、距离、地图、方位、在哪里、怎么走等空间相关问题时，在回答末尾单独一行追加「🗺️ 点击查看景区地图，了解景点分布与游览路线」

【参考资料】
{context}

【游客问题】
{question}

【导览员回答】""",
    input_variables=["context", "question", "interest_instruction"],
)

INTEREST_CONFIG = {
    "history": {
        "label": "历史文化爱好者",
        "instruction": "【游客偏好】这位游客对历史文化和佛教艺术特别感兴趣，请重点讲解景点的历史渊源、文化内涵、建筑艺术和佛教典故，用富有文化底蕴的方式介绍。\n",
    },
    "nature": {
        "label": "自然风光爱好者",
        "instruction": "【游客偏好】这位游客喜爱自然风光和户外体验，请重点介绍景点的自然景观、园林设计、观景位置和拍照打卡点，推荐最佳游览季节和时段。\n",
    },
    "family": {
        "label": "亲子家庭",
        "instruction": "【游客偏好】这位游客带着孩子出行，请用亲切活泼的语气，推荐适合亲子的景点和互动体验（如百子戏弥勒、圣水接取），注意介绍休息区和便利设施。\n",
    },
    "quick": {
        "label": "高效打卡",
        "instruction": "【游客偏好】这位游客时间有限，想高效游览精华景点，请推荐最短时间内的核心路线，简洁明了地列出必看景点和关键信息。\n",
    },
    "general": {
        "label": "自由探索",
        "instruction": "",
    },
}

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "volcano").lower()
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "BAAI/bge-small-zh-v1.5",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "")
VOLCANO_BASE_URL = os.getenv(
    "VOLCANO_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3",
)
VOLCANO_EMBEDDING_MODEL = os.getenv(
    "VOLCANO_EMBEDDING_MODEL",
    "doubao-embedding-text-240715",
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.3"))


class VolcanoVisionEmbeddings(Embeddings):
    """火山多模态 Embedding（兼容 doubao-embedding-vision 等模型）

    火山多模态 Embedding API 的 input 格式为：
        [{"type": "text", "text": "..."}, ...]
    与标准 OpenAI Embedding（["string", ...]）不同，需要自定义封装。
    """

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _call(self, texts: list[str]) -> list[list[float]]:
        """调用火山多模态 Embedding API，支持批量。"""
        url = f"{self.base_url}/embeddings/multimodal"
        payload = {
            "model": self.model,
            "input": [{"type": "text", "text": t} for t in texts],
        }
        resp = httpx.post(url, headers=self.headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        result = resp.json()
        print(f"[DEBUG] 火山 Embedding 响应 keys: {list(result.keys())}")

        raw_data = result["data"]
        if isinstance(raw_data, list):
            # 标准格式：data 是列表 [{embedding, index}, ...]
            items = sorted(raw_data, key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        elif isinstance(raw_data, dict) and "embedding" in raw_data:
            # 单条返回格式：data 直接包含 embedding
            return [raw_data["embedding"]]
        else:
            print(f"[ERROR] 未知响应格式, data type: {type(raw_data)}, 内容: {str(result)[:500]}")
            raise ValueError(f"火山 Embedding API 返回格式异常: {type(raw_data)}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]


def _get_embeddings() -> Embeddings:
    """
    创建 Embeddings：
    - volcano（默认）：火山引擎 Doubao Embedding API（OpenAI 兼容接口）
    - openai：OpenAI 兼容接口
    - local：HuggingFace 本地中文模型
    """
    if EMBEDDING_PROVIDER == "volcano":
        if not VOLCANO_API_KEY:
            raise ValueError("EMBEDDING_PROVIDER=volcano 时需设置 VOLCANO_API_KEY")
        print(f"[RAG] 使用火山 Embedding: {VOLCANO_EMBEDDING_MODEL} @ {VOLCANO_BASE_URL}")

        # 火山多模态 Embedding（如 doubao-embedding-vision）input 格式与标准 OpenAI 不同
        # 通过模型名包含 vision 或环境变量 VOLCANO_EMBEDDING_MULTIMODAL=1 启用自定义封装
        is_multimodal = (
            "vision" in VOLCANO_EMBEDDING_MODEL.lower()
            or os.getenv("VOLCANO_EMBEDDING_MULTIMODAL", "").lower() in ("1", "true", "yes")
        )
        if is_multimodal:
            print("[RAG] 检测到多模态 Embedding，使用自定义 VolcanoVisionEmbeddings")
            return VolcanoVisionEmbeddings(
                api_key=VOLCANO_API_KEY,
                base_url=VOLCANO_BASE_URL,
                model=VOLCANO_EMBEDDING_MODEL,
            )

        return OpenAIEmbeddings(
            model=VOLCANO_EMBEDDING_MODEL,
            openai_api_key=VOLCANO_API_KEY,
            openai_api_base=VOLCANO_BASE_URL,
        )

    if EMBEDDING_PROVIDER == "openai":
        if not DEEPSEEK_API_KEY:
            raise ValueError("EMBEDDING_PROVIDER=openai 时需设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")
        api_key = os.getenv("OPENAI_API_KEY", DEEPSEEK_API_KEY)
        base_url = os.getenv("OPENAI_BASE_URL", DEEPSEEK_BASE_URL)
        print(f"[RAG] 使用在线 Embedding: {EMBEDDING_MODEL} @ {base_url}")
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=api_key,
            openai_api_base=base_url,
        )

    # local：本地模型
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = LOCAL_EMBEDDING_MODEL
    if Path(model_name).is_dir():
        print(f"[RAG] 使用本地 Embedding 目录: {model_name}")
    else:
        print(f"[RAG] 使用 HuggingFace Embedding: {model_name}（首次会自动下载）")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_llm() -> ChatOpenAI:
    """创建 DeepSeek Chat 大模型客户端"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=CHAT_TEMPERATURE,
    )


def _load_documents():
    """使用 DirectoryLoader 加载 knowledge_base 下所有 .txt（UTF-8）"""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    if not any(KNOWLEDGE_DIR.glob("*.txt")):
        print(f"[警告] {KNOWLEDGE_DIR} 下暂无 .txt 文件，将创建空向量库")
        return []
    loader = DirectoryLoader(
        str(KNOWLEDGE_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load()


def build_vectordb() -> Chroma:
    """构建（或重建）向量数据库"""
    documents = _load_documents()
    embeddings = _get_embeddings()

    if not documents:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    splits = splitter.split_documents(documents)
    print(f"[RAG] 文档块数量: {len(splits)}")

    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"[RAG] 向量库已写入: {CHROMA_DIR}")
    return db


def load_vectordb() -> Chroma:
    """加载已持久化的 Chroma 向量库"""
    embeddings = _get_embeddings()
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def create_qa_chain(db: Chroma | None = None, interest: str = "general") -> RetrievalQA:
    """创建 RetrievalQA 问答链"""
    if db is None:
        db = load_vectordb()
    llm = _get_llm()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    interest_cfg = INTEREST_CONFIG.get(interest, INTEREST_CONFIG["general"])
    filled_prompt = QA_PROMPT.partial(interest_instruction=interest_cfg["instruction"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": filled_prompt},
        return_source_documents=False,
    )


def rebuild_vectordb() -> RetrievalQA:
    """重建向量库并返回新的 QA 链"""
    db = build_vectordb()
    return create_qa_chain(db)


async def astream_rag_answer(
    query: str, db: Chroma | None = None, interest: str = "general",
    image_base64: str | None = None,
) -> AsyncIterator[str]:
    """
    RAG 流式问答：先检索知识库，再逐 token 流式生成回答。
    用于 SSE 打字机效果。
    interest 支持: history, nature, family, quick, general
    image_base64: 可选，Base64 编码的图片（多模态识别）
    """
    if db is None:
        db = load_vectordb()
    llm = _get_llm()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    docs = await asyncio.to_thread(retriever.invoke, query)
    context = "\n\n".join(doc.page_content for doc in docs)
    interest_cfg = INTEREST_CONFIG.get(interest, INTEREST_CONFIG["general"])
    prompt_text = QA_PROMPT.format(
        context=context,
        question=query,
        interest_instruction=interest_cfg["instruction"],
    )

    yield f'{{"type":"expression","expression":"thinking"}}\n'

    if image_base64:
        # 多模态：构建 vision 消息格式
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        }]
        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content
    else:
        async for chunk in llm.astream(prompt_text):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content


if __name__ == "__main__":
    print("=" * 50)
    print("开始构建向量库...")
    vector_db = build_vectordb()
    print("构建完成，正在创建 QA 链...")
    qa = create_qa_chain(vector_db)
    test_q = "景区开放时间是几点？"
    print(f"\n测试问题: {test_q}")
    try:
        answer = qa.invoke({"query": test_q})
        print(f"测试回答: {answer.get('result', answer)}")
    except Exception as e:
        print(f"测试问答失败: {e}")
    print("=" * 50)
