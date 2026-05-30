#!/usr/bin/env python3
"""
AI/LLM Agent 知识库扩写 Worker
每3分钟自动扩写一个知识点，基于进度文件选择最浅的topic进行深化。
"""

import json
import os
import subprocess
import time
import re
from datetime import datetime
from pathlib import Path

# ── 配置 ──
BASE_DIR = Path.home() / "learn/ai-agent-learning/ai-llm-agent-knowledge-base"
NOTES_FILE = Path.home() / "learn/ai-agent-learning/AI-LLM-Agent 面试学习笔记.md"
PROGRESS_FILE = BASE_DIR / "expansion_progress.json"
INTERVAL_SECONDS = 180  # 3分钟

# 模型API配置 - 使用 dashscope (通义千问)
import urllib.request
import urllib.error

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_BASE = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("EXPAND_MODEL", "qwen-plus")

# 章节结构
CHAPTERS = {
    "01-学习路线与岗位介绍": [
        "大模型岗位分类与职责", "算法工程师vs应用工程师",
        "自学路线与资源推荐", "技能树与能力模型"
    ],
    "02-LLM模型结构与训练": [
        "Transformer架构详解", "编码器与解码器的区别", "预训练阶段",
        "SFT监督微调", "RLHF与对齐训练", "千问模型演进",
        "DeepSeek模型特点", "GPT系列模型分析"
    ],
    "03-模型与硬件": [
        "模型内存需求计算", "显存与计算资源评估", "CPU推理方案",
        "GPU选型建议", "不同硬件推理对比"
    ],
    "04-模型对比与选择": [
        "主流模型横向对比", "模型选型方法论", "开源vs闭源模型",
        "模型评测基准", "场景化选型实践"
    ],
    "05-RAG检索增强生成": [
        "RAG核心流程概览", "文档切分策略", "Embedding模型选择",
        "向量数据库对比", "检索策略与召回优化", "重排序方法",
        "RAG评估指标RAGAS", "Agentic-RAG", "Graph-RAG", "Hybrid-Search混合检索"
    ],
    "06-Agent开发框架": [
        "LangChain框架详解", "LangGraph状态机", "Semantic-Kernel框架",
        "AutoGPT与自主Agent", "Function-Calling机制", "ReAct模式",
        "工具调用失败处理", "Agent设计模式"
    ],
    "07-Multi-Agent系统": [
        "单Agentvs多Agent", "中心化架构设计", "去中心化架构",
        "层级式架构", "Agent通信机制", "状态管理与共享",
        "错误传播与控制", "多Agent场景设计"
    ],
    "08-微调与训练": [
        "LoRA原理与实践", "数据集准备与标注", "超参数调优",
        "SFT训练流程", "DPO直接偏好优化", "强化学习对齐", "微调实验方法论"
    ],
    "09-推理加速与量化": [
        "量化技术概述", "INT4-INT8量化方案", "KV-Cache机制",
        "推理引擎对比", "ONNX优化", "推理加速实践方案"
    ],
    "10-MCP与A2A协议": [
        "MCP协议架构", "MCP本地vs远端Server", "MCP版本协商与能力协商",
        "A2A协议详解", "MCP与A2A对比与选型", "MCP工具开发实践"
    ],
    "11-提示词工程": [
        "提示词设计原则", "提示词压缩与裁剪", "数据飞轮优化",
        "COT思维链", "Few-shot与In-context-learning", "结构化输出约束"
    ],
    "12-Agent评估与监控": [
        "评估维度与指标体系", "LLM-as-Judge方法", "轨迹评估",
        "工具调用评估", "线上监控方案", "离线评估数据集设计", "美团龙猫评估方法"
    ],
    "13-记忆机制": [
        "短期记忆与上下文窗口", "长期记忆设计", "上下文压缩策略",
        "用户画像与个性化", "记忆持久化方案"
    ],
    "14-幻觉问题": [
        "幻觉的定义与分类", "幻觉产生的根因",
        "减少幻觉的工程方法", "幻觉检测与兜底策略"
    ],
    "15-架构设计模式": [
        "Agent系统架构模板", "事件驱动架构", "OpenClaw三层架构",
        "ClaudeCode设计思想", "场景化架构设计"
    ],
    "16-面试技巧与实战": [
        "自我介绍技巧", "项目包装与深度挖掘", "反问技巧",
        "意向度展示", "常见面试问题应对", "谈薪策略"
    ],
}


def load_progress():
    with open(PROGRESS_FILE, "r") as f:
        return json.load(f)


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def find_next_topic(progress):
    """找到expansion_level最低的topic，优先选level=0的"""
    candidates = []
    for chapter, ch_data in progress["chapters"].items():
        for topic, tdata in ch_data["topics"].items():
            level = tdata.get("expansion_level", 0)
            rounds = tdata.get("rounds_spent", 0)
            candidates.append((level, rounds, chapter, topic))

    # 按level升序，同level按rounds升序
    candidates.sort(key=lambda x: (x[0], x[1]))

    if candidates and candidates[0][0] < 4:  # 最高扩展到level 4
        _, _, chapter, topic = candidates[0]
        return chapter, topic
    return None, None


def topic_to_filename(topic):
    return topic.replace(" ", "-").replace("/", "-").lower() + ".md"


def read_existing_content(chapter, topic):
    """读取已有内容"""
    filepath = BASE_DIR / chapter / topic_to_filename(topic)
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def read_notes_context(topic):
    """从原始笔记中提取相关内容"""
    if not NOTES_FILE.exists():
        return ""
    try:
        content = NOTES_FILE.read_text(encoding="utf-8")
        # 搜索包含该topic关键词的段落
        lines = content.split("\n")
        relevant = []
        for i, line in enumerate(lines):
            # 简单的关键词匹配
            keywords = topic.replace("vs", " ").replace("-", " ").split()
            if any(kw.lower() in line.lower() for kw in keywords if len(kw) > 2):
                start = max(0, i - 2)
                end = min(len(lines), i + 15)
                relevant.append("\n".join(lines[start:end]))
        # 限制长度
        combined = "\n---\n".join(relevant[:5])
        return combined[:3000]
    except Exception:
        return ""


def call_llm(system_prompt, user_prompt, max_tokens=8000):
    """调用LLM API生成内容"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", API_KEY)
    url = f"{API_BASE}/chat/completions"
    data = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"  [ERROR] API返回 {e.code}: {error_body[:200]}")
        return None
    except Exception as e:
        print(f"  [ERROR] API调用失败: {e}")
        return None


def generate_expansion(chapter, topic, existing_content, notes_context, progress):
    """生成扩写内容"""
    round_num = progress["completed_rounds"] + 1
    current_level = 0
    try:
        current_level = progress["chapters"][chapter]["topics"][topic].get("expansion_level", 0)
    except KeyError:
        pass

    system_prompt = """你是一位资深的AI/LLM Agent技术专家，拥有多年工业级Agent系统开发经验。
你需要为技术知识库撰写高质量的深度学习文档。

要求：
- 内容必须是真实可验证的技术知识，不要编造API或不存在的功能
- 代码示例用Python，标注依赖版本
- 包含工业界最佳实践和踩坑经验
- 面试相关问题要基于真实面试场景
- 每个文件至少2000字
- 使用清晰的markdown结构"""

    if current_level == 0 and not existing_content:
        # 首次生成 - 完整文档
        user_prompt = f"""请为以下知识点撰写完整的技术文档。

章节：{chapter}
知识点：{topic}

原始笔记中的相关内容：
{notes_context[:2000] if notes_context else '（无直接相关内容，请基于专业知识生成）'}

请生成一份完整的技术文档，包含以下结构：

# {topic}

## 1. 核心概念与原理
（详细解释该技术的本质、设计思想）

## 2. 技术细节与实现机制
（深入讲解内部工作原理、关键算法、数据流）

## 3. 代码示例
（可运行的Python代码，标注依赖版本）

## 4. 工业界最佳实践
（大厂真实项目中的做法、架构选型）

## 5. 常见面试问题与参考答案
（至少5个高频面试题，给出详细答案）

## 6. 优缺点对比
（表格形式对比不同方案）

## 7. 与其他技术的关系
（和相近技术的对比、互补关系）

## 8. 踩坑经验与注意事项
（实际开发中容易犯的错误、性能陷阱）

## 9. 参考资料
（官方文档链接、论文、开源项目）

请确保内容深度足够，面向有1-2年经验的开发者。"""

    elif current_level >= 1 and existing_content:
        # 深化已有内容
        user_prompt = f"""以下是一份已有的技术文档，请对其进行深化扩写。

章节：{chapter}
知识点：{topic}
当前深度级别：{current_level}/4

已有内容（前2000字）：
{existing_content[:2000]}

原始笔记补充：
{notes_context[:1500] if notes_context else '无'}

请在已有内容基础上，补充以下方面的深度内容（选择2-3个方向深化）：

1. **更多工业案例**：大厂（字节/阿里/美团/OpenAI/Anthropic）的真实实践
2. **性能调优细节**：具体的benchmark数据、调优前后的对比
3. **高级设计模式**：进阶架构、复杂场景处理
4. **面试深度追问**：面试官层层追问的连环问题和应对
5. **源码级理解**：核心库的源码解析、关键函数说明
6. **前沿论文解读**：最新研究成果对该技术的影响

请输出完整的新版本文档（替换原有内容），确保比原版更深入、更全面。
不要输出"在原文基础上增加"这类说明，直接输出完整文档。"""
    else:
        user_prompt = f"""请为以下知识点撰写技术文档。

章节：{chapter}
知识点：{topic}

相关内容：
{notes_context[:2000] if notes_context else '无直接相关内容'}

请生成结构化的技术文档，重点包含原理、实践、面试题目。"""

    return call_llm(system_prompt, user_prompt)


def git_commit(chapter, topic, round_num):
    """Git提交"""
    try:
        subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, capture_output=True, timeout=10)
        msg = f"第{round_num}轮扩写: {chapter}/{topic}"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=BASE_DIR, capture_output=True, timeout=10
        )
        return True
    except Exception as e:
        print(f"  [WARN] Git commit failed: {e}")
        return False


def run_one_round():
    """执行一轮扩写"""
    progress = load_progress()
    round_num = progress["completed_rounds"] + 1

    if round_num > 3000:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 已达3000轮上限，停止。")
        return False

    chapter, topic = find_next_topic(progress)
    if not chapter:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 所有topic已扩写到最高级别。")
        return False

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 第{round_num}轮: {chapter}/{topic}")

    # 读取已有内容和笔记上下文
    existing = read_existing_content(chapter, topic)
    notes = read_notes_context(topic)
    current_level = progress["chapters"].get(chapter, {}).get("topics", {}).get(topic, {}).get("expansion_level", 0)
    print(f"  当前级别: {current_level}, 已有内容: {len(existing)}字")

    # 生成内容
    print(f"  调用 {MODEL} 生成内容...")
    content = generate_expansion(chapter, topic, existing, notes, progress)
    if not content:
        print(f"  [FAIL] 生成失败，跳过")
        return True  # 继续下一轮

    # 写入文件
    filepath = BASE_DIR / chapter / topic_to_filename(topic)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    print(f"  写入: {filepath} ({len(content)}字)")

    # 更新进度
    new_level = min(current_level + 1, 4)
    progress["completed_rounds"] = round_num
    progress["chapters"][chapter]["topics"][topic]["status"] = "expanded" if new_level < 3 else "deep_expanded"
    progress["chapters"][chapter]["topics"][topic]["expansion_level"] = new_level
    progress["chapters"][chapter]["topics"][topic]["rounds_spent"] = \
        progress["chapters"][chapter]["topics"][topic].get("rounds_spent", 0) + 1
    progress["chapters"][chapter]["topics"][topic]["last_updated"] = datetime.now().isoformat()
    save_progress(progress)

    # Git提交
    git_commit(chapter, topic, round_num)
    print(f"  ✓ 完成 (级别 {current_level}→{new_level})")
    return True


def main():
    print("=" * 60)
    print("AI/LLM Agent 知识库扩写 Worker")
    print(f"模型: {MODEL}")
    print(f"API: {API_BASE}")
    print(f"间隔: {INTERVAL_SECONDS}秒")
    print(f"目录: {BASE_DIR}")
    print("=" * 60)

    if not API_KEY:
        print("[ERROR] 未找到 DASHSCOPE_API_KEY 环境变量！")
        print("请在 ~/.hermes/.env 或环境变量中设置")
        return

    # 读取API key (从.env文件加载)
    env_file = Path.home() / ".hermes" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DASHSCOPE_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    os.environ["DASHSCOPE_API_KEY"] = key

    progress = load_progress()
    print(f"当前进度: {progress['completed_rounds']}/3000")
    print()

    while True:
        try:
            should_continue = run_one_round()
            if not should_continue:
                break
        except KeyboardInterrupt:
            print("\n收到中断信号，退出。")
            break
        except Exception as e:
            print(f"[ERROR] 本轮异常: {e}")

        print(f"  等待 {INTERVAL_SECONDS}秒...\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
