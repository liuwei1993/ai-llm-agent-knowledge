#!/usr/bin/env python3
"""单次执行一轮扩写，供cronjob no_agent模式调用"""
import json, os, sys, subprocess, time
from datetime import datetime
from pathlib import Path
import urllib.request, urllib.error

BASE_DIR = Path.home() / "learn/ai-agent-learning/ai-llm-agent-knowledge-base"
NOTES_FILE = Path.home() / "learn/ai-agent-learning/AI-LLM-Agent 面试学习笔记.md"
PROGRESS_FILE = BASE_DIR / "expansion_progress.json"
LOG_FILE = BASE_DIR / "expand_worker.log"

# 加载环境变量
env_file = Path.home() / ".hermes" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_BASE = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = "qwen-plus"

CHAPTERS = {
    "01-学习路线与岗位介绍": ["大模型岗位分类与职责", "算法工程师vs应用工程师", "自学路线与资源推荐", "技能树与能力模型"],
    "02-LLM模型结构与训练": ["Transformer架构详解", "编码器与解码器的区别", "预训练阶段", "SFT监督微调", "RLHF与对齐训练", "千问模型演进", "DeepSeek模型特点", "GPT系列模型分析"],
    "03-模型与硬件": ["模型内存需求计算", "显存与计算资源评估", "CPU推理方案", "GPU选型建议", "不同硬件推理对比"],
    "04-模型对比与选择": ["主流模型横向对比", "模型选型方法论", "开源vs闭源模型", "模型评测基准", "场景化选型实践"],
    "05-RAG检索增强生成": ["RAG核心流程概览", "文档切分策略", "Embedding模型选择", "向量数据库对比", "检索策略与召回优化", "重排序方法", "RAG评估指标RAGAS", "Agentic-RAG", "Graph-RAG", "Hybrid-Search混合检索"],
    "06-Agent开发框架": ["LangChain框架详解", "LangGraph状态机", "Semantic-Kernel框架", "AutoGPT与自主Agent", "Function-Calling机制", "ReAct模式", "工具调用失败处理", "Agent设计模式"],
    "07-Multi-Agent系统": ["单Agentvs多Agent", "中心化架构设计", "去中心化架构", "层级式架构", "Agent通信机制", "状态管理与共享", "错误传播与控制", "多Agent场景设计"],
    "08-微调与训练": ["LoRA原理与实践", "数据集准备与标注", "超参数调优", "SFT训练流程", "DPO直接偏好优化", "强化学习对齐", "微调实验方法论"],
    "09-推理加速与量化": ["量化技术概述", "INT4-INT8量化方案", "KV-Cache机制", "推理引擎对比", "ONNX优化", "推理加速实践方案"],
    "10-MCP与A2A协议": ["MCP协议架构", "MCP本地vs远端Server", "MCP版本协商与能力协商", "A2A协议详解", "MCP与A2A对比与选型", "MCP工具开发实践"],
    "11-提示词工程": ["提示词设计原则", "提示词压缩与裁剪", "数据飞轮优化", "COT思维链", "Few-shot与In-context-learning", "结构化输出约束"],
    "12-Agent评估与监控": ["评估维度与指标体系", "LLM-as-Judge方法", "轨迹评估", "工具调用评估", "线上监控方案", "离线评估数据集设计", "美团龙猫评估方法"],
    "13-记忆机制": ["短期记忆与上下文窗口", "长期记忆设计", "上下文压缩策略", "用户画像与个性化", "记忆持久化方案"],
    "14-幻觉问题": ["幻觉的定义与分类", "幻觉产生的根因", "减少幻觉的工程方法", "幻觉检测与兜底策略"],
    "15-架构设计模式": ["Agent系统架构模板", "事件驱动架构", "OpenClaw三层架构", "ClaudeCode设计思想", "场景化架构设计"],
    "16-面试技巧与实战": ["自我介绍技巧", "项目包装与深度挖掘", "反问技巧", "意向度展示", "常见面试问题应对", "谈薪策略"],
}

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_progress():
    with open(PROGRESS_FILE) as f:
        return json.load(f)

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

def find_next(progress):
    cands = []
    for ch, cd in progress["chapters"].items():
        for t, td in cd["topics"].items():
            cands.append((td.get("expansion_level", 0), td.get("rounds_spent", 0), ch, t))
    cands.sort(key=lambda x: (x[0], x[1]))
    if cands and cands[0][0] < 4:
        return cands[0][2], cands[0][3]
    return None, None

def topic_fn(topic):
    return topic.replace(" ", "-").replace("/", "-").lower() + ".md"

def read_existing(ch, t):
    p = BASE_DIR / ch / topic_fn(t)
    return p.read_text() if p.exists() else ""

def read_notes(topic):
    if not NOTES_FILE.exists():
        return ""
    content = NOTES_FILE.read_text()
    lines = content.split("\n")
    kws = [w for w in topic.replace("vs", " ").replace("-", " ").split() if len(w) > 2]
    relevant = []
    for i, line in enumerate(lines):
        if any(kw.lower() in line.lower() for kw in kws):
            s, e = max(0, i-2), min(len(lines), i+15)
            relevant.append("\n".join(lines[s:e]))
    return "\n---\n".join(relevant[:5])[:3000]

def call_llm(sys_p, usr_p):
    data = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": usr_p}
    ], "max_tokens": 8000, "temperature": 0.3}).encode()
    req = urllib.request.Request(f"{API_BASE}/chat/completions", data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        log(f"  API错误: {e}")
        return None

def run_round():
    progress = load_progress()
    rn = progress["completed_rounds"] + 1
    if rn > 3000:
        log("已达3000轮上限")
        return

    ch, topic = find_next(progress)
    if not ch:
        log("所有topic已达最高级别")
        return

    log(f"第{rn}轮: {ch}/{topic}")
    existing = read_existing(ch, topic)
    notes = read_notes(topic)
    cl = progress["chapters"].get(ch, {}).get("topics", {}).get(topic, {}).get("expansion_level", 0)

    sys_p = "你是资深AI/LLM Agent技术专家，拥有多年工业级Agent系统开发经验。为技术知识库撰写高质量深度学习文档。内容必须真实可验证，代码用Python标注版本，包含工业最佳实践和踩坑经验，面试问题基于真实场景，每文件至少2000字，markdown结构清晰。"

    if cl == 0 and not existing:
        usr_p = f"为知识点「{topic}」（章节：{ch}）撰写完整技术文档。\n\n原始笔记相关内容：\n{notes[:2000] if notes else '无直接相关内容'}\n\n结构：# {topic}\n## 1. 核心概念与原理\n## 2. 技术细节与实现机制\n## 3. 代码示例（Python可运行）\n## 4. 工业界最佳实践\n## 5. 常见面试问题与参考答案（至少5题）\n## 6. 优缺点对比（表格）\n## 7. 与其他技术的关系\n## 8. 踩坑经验与注意事项\n## 9. 参考资料\n\n面向有1-2年经验的开发者。"
    else:
        usr_p = f"深化扩写「{topic}」（章节：{ch}，当前级别{cl}/4）。\n\n已有内容前2000字：\n{existing[:2000]}\n\n补充方向（选2-3个）：\n1. 更多工业案例（字节/阿里/美团/OpenAI/Anthropic）\n2. 性能调优benchmark数据\n3. 高级设计模式与复杂场景\n4. 面试深度追问连环题\n5. 源码级解析\n6. 前沿论文解读\n\n输出完整新版文档替换原版，不要说明性文字。"

    log(f"  调用{MODEL}生成...")
    content = call_llm(sys_p, usr_p)
    if not content:
        log("  生成失败，跳过")
        return

    fp = BASE_DIR / ch / topic_fn(topic)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    log(f"  写入: {fp.name} ({len(content)}字)")

    nl = min(cl + 1, 4)
    progress["completed_rounds"] = rn
    progress["chapters"][ch]["topics"][topic]["status"] = "expanded" if nl < 3 else "deep_expanded"
    progress["chapters"][ch]["topics"][topic]["expansion_level"] = nl
    progress["chapters"][ch]["topics"][topic]["rounds_spent"] = progress["chapters"][ch]["topics"][topic].get("rounds_spent", 0) + 1
    progress["chapters"][ch]["topics"][topic]["last_updated"] = datetime.now().isoformat()
    save_progress(progress)

    try:
        subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, capture_output=True, timeout=10)
        subprocess.run(["git", "commit", "-m", f"第{rn}轮扩写: {ch}/{topic}"],
                       cwd=BASE_DIR, capture_output=True, timeout=10)
    except:
        pass
    log(f"  ✓ 完成 (级别{cl}→{nl})，进度: {rn}/3000")

if __name__ == "__main__":
    run_round()
