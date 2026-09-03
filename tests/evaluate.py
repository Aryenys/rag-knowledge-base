"""
【W5】评估脚本：量化检索效果，产出 README 和简历用的对比数据。
方法：准备 20 条测试问题，每条标注"正确出处 chunk 的关键词"，
     跑检索链路，统计 命中率 = 正确 chunk 出现在 top-K 中的比例。
对比维度（改 config 参数各跑一遍）：
  ① 纯向量召回 top5
  ② 混合召回 top5（向量+BM25）
  ③ 混合召回 + rerank top5
  ④ 不同 chunk_size（300 / 500 / 800）
输出：一个 markdown 表格，直接贴进 README。
用法：python -m tests.evaluate
"""

# 测试集示例格式（W5 时针对你上传的文档编写）：
TEST_CASES = [
    # {"question": "...", "expected_keyword": "该问题答案所在 chunk 里的一个特征词"},
]


def main():
    # TODO(W5):
    # for case in TEST_CASES:
    #     chunks = 检索链路(case["question"])
    #     hit = any(case["expected_keyword"] in c["content"] for c in chunks)
    # 汇总命中率，打印 markdown 表格
    raise NotImplementedError("W5 实现")


if __name__ == "__main__":
    main()
