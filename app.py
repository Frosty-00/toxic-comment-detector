# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json
import re

app = FastAPI()

# 设置模板目录 (寻找前端网页)
templates = Jinja2Templates(directory="templates")

print("正在唤醒 AI 引擎...")
# 模型搜索规则：
# 1) 优先使用 ./results/final-model
# 2) 若不存在，则自动选择 ./results/final-model-* 中最新且包含权重的目录
# 3) 若仍不存在，则自动选择 ./results/checkpoint-* 中编号最大的且包含权重的目录
MODEL_PATH = "./results/final-model"
RESULTS_DIR = "./results"
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "thresholds.json")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

model = None
tokenizer = None
# 默认走 CPU，避免部分显卡与当前 PyTorch CUDA 版本不兼容导致 500 错误。
# 如需尝试 GPU，可在启动前设置环境变量 USE_CUDA=1。
device = torch.device("cpu")
if os.getenv("USE_CUDA", "0") == "1" and torch.cuda.is_available():
    device = torch.device("cuda")
model_error = None
loaded_model_path = None
thresholds = {label: 0.5 for label in LABELS}

LEET_TABLE = str.maketrans({
    "@": "a",
    "$": "s",
    "!": "i",
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
})

ABBR_MAP = {
    "fk": "fuck",
    "fkk": "fuck",
    "fuk": "fuck",
    "fck": "fuck",
    "wtf": "what the fuck",
    "stfu": "shut the fuck up",
    "btch": "bitch",
    "trsh": "trash",
}

def _has_model_weights(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "model.safetensors"))
        or os.path.exists(os.path.join(path, "pytorch_model.bin"))
    )

def _select_model_path() -> str:
    if os.path.exists(MODEL_PATH) and _has_model_weights(MODEL_PATH):
        return MODEL_PATH

    if not os.path.exists(RESULTS_DIR):
        raise FileNotFoundError(f"结果目录不存在: {RESULTS_DIR}")

    final_candidates = []
    for name in os.listdir(RESULTS_DIR):
        full_path = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith("final-model-"):
            continue
        if _has_model_weights(full_path):
            final_candidates.append(full_path)

    if final_candidates:
        final_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return final_candidates[0]

    candidates = []
    for name in os.listdir(RESULTS_DIR):
        full_path = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith("checkpoint-"):
            continue

        suffix = name.replace("checkpoint-", "", 1)
        if not suffix.isdigit():
            continue
        if _has_model_weights(full_path):
            candidates.append((int(suffix), full_path))

    if not candidates:
        raise FileNotFoundError(
            "未找到可用模型：./results/final-model 不可用，final-model-* 不可用，且 checkpoint-* 中没有权重文件 "
            "(需要 model.safetensors 或 pytorch_model.bin)"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def _load_thresholds():
    if not os.path.exists(THRESHOLD_PATH):
        print("未找到 thresholds.json，使用默认阈值 0.5。")
        return
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for label in LABELS:
            value = loaded.get(label, 0.5)
            thresholds[label] = float(value)
        print(f"已加载每标签阈值: {thresholds}")
    except Exception as e:
        print(f"阈值文件读取失败，使用默认阈值 0.5。错误: {e}")

def _normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.translate(LEET_TABLE)
    # 去掉混淆用符号，保留字母数字与空格
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    tokens = normalized.split()
    tokens = [ABBR_MAP.get(token, token) for token in tokens]
    return " ".join(tokens)

try:
    loaded_model_path = _select_model_path()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(loaded_model_path)
    model.to(device)
    model.eval()
    _load_thresholds()
    print(f"AI 引擎就绪！已加载模型: {loaded_model_path}")
except Exception as e:
    model_error = str(e)
    print(f"模型加载失败，请检查路径: {model_error}")

# 定义接收数据的格式
class ChatInput(BaseModel):
    text: str

@app.get("/api/health")
async def health_check():
    ready = model is not None and tokenizer is not None
    return {
        "ok": ready,
        "ready": ready,
        "model_path": loaded_model_path if ready else None,
        "device": str(device),
        "thresholds": thresholds,
        "error": None if ready else model_error,
    }

# API 接口：接收前端发来的聊天文本，返回 6 个维度的恶意得分
@app.post("/api/analyze")
async def analyze_text(chat_input: ChatInput):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "模型未就绪，请先检查模型路径与权重文件。"
                f"优先路径: {MODEL_PATH}；结果目录: {RESULTS_DIR}；错误: {model_error}"
            ),
        )

    text = chat_input.text
    normalized_text = _normalize_text(text)
    inputs = tokenizer(normalized_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits).cpu().numpy()[0]
    
    # 将结果打包成 JSON 返回给前端网页
    result = {label: float(prob) for label, prob in zip(LABELS, probs)}
    
    # 按每个标签的专属阈值判定，避免稀有标签被统一 0.5 压制
    hits = {label: result[label] > thresholds.get(label, 0.5) for label in LABELS}
    is_toxic = any(hits.values())
    return {
        "ok": True,
        "is_toxic": is_toxic,
        "scores": result,
        "hits": hits,
        "thresholds": thresholds,
        "normalized_text": normalized_text,
    }

# 提供前端网页
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})