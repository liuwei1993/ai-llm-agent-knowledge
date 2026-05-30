#!/bin/bash
# 扩写任务启动脚本
cd ~/learn/ai-agent-learning/ai-llm-agent-knowledge-base

# 加载环境变量
if [ -f ~/.hermes/.env ]; then
    set -a
    source <(grep -E '^[A-Z].*=' ~/.hermes/.env | grep -v '^#')
    set +a
fi

export PYTHONUNBUFFERED=1
exec python3 -u expand_worker.py >> expand_worker.log 2>&1
