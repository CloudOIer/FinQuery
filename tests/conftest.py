"""pytest 全局配置。

FINQUERY_LLM_CONFIG 指向不存在的文件 → load_llm_settings() 返回全禁用的
LLMSettings → HybridIntentEngine 静默走规则引擎、答案润色走确定性文案。

为什么需要这层隔离:api/main.py 在 import 时装配 HybridIntentEngine,若读到
真实 config/llm.json(intent_enabled=true),所有 API 测试都会真实调用
DeepSeek —— 单测会变慢(秒级网络往返)、消耗配额、且在断网/CI 环境不稳定。
单测只验证代码逻辑,LLM 效果由评测脚本负责(data/evaluation/)。

必须在 import finquery_agent 之前设置,故放在 conftest 顶层而不是 fixture 里
(pytest 收集测试文件时就会触发 api/main.py 的模块级装配)。
"""

import os

os.environ["FINQUERY_LLM_CONFIG"] = os.path.join(os.path.dirname(__file__), "llm.disabled.json")
