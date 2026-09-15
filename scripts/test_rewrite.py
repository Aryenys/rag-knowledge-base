"""Step 12 查询改写验证脚本"""
from app.rag.generator import rewrite_query

# 场景 1：有历史，需要指代消解
history = [
    {"role": "user", "content": "厦航奖学金的奖励标准是什么"},
    {"role": "assistant", "content": "厦航奖学金奖励全日制在校本科生及研究生，金额为每人5000元。"},
]
q1 = "它什么时候截止申请"
print(f"原始问题: {q1}")
print(f"改写结果: {rewrite_query(q1, history)}")
print("-" * 50)

# 场景 2：空历史，应该原样返回（无 LLM 调用）
q2 = "你好"
print(f"原始问题: {q2}")
print(f"改写结果: {rewrite_query(q2, [])}")
print("-" * 50)

# 场景 3：更复杂的指代
history3 = [
    {"role": "user", "content": "BGE-M3 是什么"},
    {"role": "assistant", "content": "BGE-M3 是智源研究院开源的多语言 embedding 模型。"},
]
q3 = "它支持哪些检索方式"
print(f"原始问题: {q3}")
print(f"改写结果: {rewrite_query(q3, history3)}")
