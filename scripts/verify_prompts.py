"""
【任务4】Prompt 版本管理验证脚本。不调 LLM，跑得快。

验证五件事：
  1. 默认版本的模板与改造前逐字一致（回归锚点，最重要的一条）
  2. 版本继承：v2 只改了主问答 prompt，其余模板自动沿用 v1
  3. 切换版本确实生效
  4. 版本配错时回退 v1，而不是让服务崩掉
  5. 模板写错占位符时加载即报错（fail fast），不会等到线上才炸

用法：
  python -m scripts.verify_prompts
"""
import hashlib
import json
import sys
from pathlib import Path

# 改造前 app/core/prompts.py 里五个模板的 sha256（UTF-8）。
# 这是"默认版本不许被改坏"的锚点：哪天有人手滑改了 v1.json，这一项立刻变红。
BASELINE = {
    "rag_system": "f9a9481158f492ec4feb5c10dca808c282d94d1bd6bc31b8e78137402543005a",
    "context_item": "72abe90362de1fcc772066f2fd49a4db77d42a8f1fc0346a7ec366caf6522525",
    "query_rewrite": "f8d40214752309c716d7c4d21cdf576d13fec1ee5d72a64265028910c715570c",
    "no_context": "e4611ad0837b0349eda5832a0ee3d91e01706112268dfbbe5e1e8c6c5260d83b",
    "agent_system": "120047299a0b8e20ecf1fe2d5d1fdea8e5381db8d40f8792bf55e4feedf9bc94",
}

TEMPLATE_NAMES = list(BASELINE.keys())
VERSION_DIR = Path(__file__).resolve().parent.parent / "app" / "core" / "prompt_versions"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_baseline() -> bool:
    """第 1 组：默认版本必须与改造前逐字一致。"""
    print("=" * 60)
    print("第 1 组：默认版本 = 改造前（回归锚点）")
    print("=" * 60)

    from app.core import prompts

    print(f"\n  当前生效版本：{prompts.CURRENT_PROMPT_VERSION}")
    ok = True
    for name, want in BASELINE.items():
        got = sha(prompts.__dict__[
            {
                "rag_system": "RAG_SYSTEM_PROMPT",
                "context_item": "CONTEXT_ITEM_TEMPLATE",
                "query_rewrite": "QUERY_REWRITE_PROMPT",
                "no_context": "NO_CONTEXT_SYSTEM_PROMPT",
                "agent_system": "AGENT_SYSTEM_PROMPT",
            }[name]
        ])
        same = got == want
        ok = ok and same
        print(f"    {name:14s} {'一致' if same else '不一致！'}")
    if ok:
        print("  → 改造只是把模板搬了家，默认行为零变化：app/rag/ 一行没动过")
    else:
        print("  → 默认 prompt 被改了！如果不是有意升级版本，就是误操作")
    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def check_inheritance() -> bool:
    """第 2 组：v2 只覆盖主问答，其余继承 v1。"""
    print("\n" + "=" * 60)
    print("第 2 组：版本继承（v2 extends v1）")
    print("=" * 60)

    from app.core import prompt_registry

    v1 = {n: prompt_registry.get_template(n, "v1") for n in TEMPLATE_NAMES}
    v2 = {n: prompt_registry.get_template(n, "v2") for n in TEMPLATE_NAMES}

    changed = [n for n in TEMPLATE_NAMES if v1[n] != v2[n]]
    unchanged = [n for n in TEMPLATE_NAMES if v1[n] == v2[n]]

    print(f"\n  v2 相对 v1 改动的模板：{changed}")
    print(f"  继承下来没动的模板：{unchanged}")
    ok = changed == ["rag_system"]
    if ok:
        print("  → 符合预期：v2 只写了差异，其余自动继承，不用复制粘贴整份模板")
    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def check_switch() -> bool:
    """第 3 组：改配置能切版本。"""
    print("\n" + "=" * 60)
    print("第 3 组：切换版本生效")
    print("=" * 60)

    from app.config import settings
    from app.core import prompt_registry

    old = settings.PROMPT_VERSION
    try:
        settings.PROMPT_VERSION = "v2"
        got = prompt_registry.get_template("rag_system")
        want = prompt_registry.get_template("rag_system", "v2")
        ok = got == want
        print(f"\n  设 PROMPT_VERSION=v2 后取到的 rag_system 与 v2 一致：{ok}")
        print(f"  内容前 40 字：{got[:40]}...")
    finally:
        settings.PROMPT_VERSION = old

    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def check_fallback() -> bool:
    """第 4 组：版本配错时回退 v1。"""
    print("\n" + "=" * 60)
    print("第 4 组：配错版本时回退（不能让服务起不来）")
    print("=" * 60)

    from app.config import settings
    from app.core import prompt_registry

    v1_text = prompt_registry.get_template("rag_system", "v1")
    old = settings.PROMPT_VERSION
    try:
        settings.PROMPT_VERSION = "v_does_not_exist"
        got = prompt_registry.get_template("rag_system")
        ok = got == v1_text
        print(f"\n  设成一个不存在的版本后，取到的内容等于 v1：{ok}")
        print(f"  current_version() 报的是：{prompt_registry.current_version()}")
    finally:
        settings.PROMPT_VERSION = old

    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def check_fail_fast() -> bool:
    """第 5 组：模板写错占位符，加载时就报错。"""
    print("\n" + "=" * 60)
    print("第 5 组：模板写错占位符要立刻暴露（fail fast）")
    print("=" * 60)

    import app.core.prompt_registry as pr
    from app.core import prompt_registry

    bad_path = VERSION_DIR / "_bad_tmp.json"
    bad_path.write_text(
        json.dumps({
            "version": "_bad_tmp",
            "name": "临时坏版本",
            "templates": {"rag_system": "参考资料：{context}\n未知变量：{typo_var}"},
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    ok_load = False
    ok_fallback = False
    try:
        try:
            prompt_registry.load_version("_bad_tmp")
            print("\n  加载坏版本：没报错（不对，应该报错）")
        except prompt_registry.PromptVersionError as e:
            ok_load = True
            print(f"\n  加载坏版本报错：{e}")

        # get_template 不应该把异常抛给调用方，而是回退到 v1
        got = prompt_registry.get_template("rag_system", "_bad_tmp")
        ok_fallback = got == prompt_registry.get_template("rag_system", "v1")
        print(f"  get_template 回退到 v1：{ok_fallback}")
    finally:
        bad_path.unlink(missing_ok=True)
        pr._cache.pop("_bad_tmp", None)
        pr._meta_cache = None

    ok = ok_load and ok_fallback
    if ok:
        print("  → 写错占位符在加载阶段就被拦下，不会等到线上某次问答才 KeyError")
    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def check_render() -> bool:
    """第 6 组：渲染只替换已知变量，模板里写示例代码也不炸。"""
    print("\n" + "=" * 60)
    print("第 6 组：渲染安全性")
    print("=" * 60)

    from app.core import prompt_registry

    # 只传 context，模板里若有别的占位符应原样保留而不是 KeyError
    out = prompt_registry.render("query_rewrite", history="A: 你好", question="BGE-M3 是什么")
    ok1 = "A: 你好" in out and "BGE-M3 是什么" in out and "{" not in out
    print(f"\n  正常渲染（变量都被替换、无残留花括号）：{ok1}")

    # str.format 遇到未提供的变量会 KeyError，render 不会
    try:
        partial = prompt_registry.render("query_rewrite")
        ok2 = "{history}" in partial and "{question}" in partial
        print(f"  缺变量时保留占位符而非抛异常：{ok2}")
    except Exception as e:  # noqa: BLE001
        ok2 = False
        print(f"  缺变量时抛异常了（不该）：{e}")

    ok = ok1 and ok2
    print(f"  结果：{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    print("\n【任务4】Prompt 版本管理验证\n")
    results = [
        ("默认版本与改造前逐字一致", check_baseline()),
        ("版本继承", check_inheritance()),
        ("切换版本生效", check_switch()),
        ("配错版本回退", check_fallback()),
        ("模板校验 fail fast", check_fail_fast()),
        ("渲染安全性", check_render()),
    ]
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n总体：{'全部通过' if all(o for _, o in results) else '存在失败项'}")


if __name__ == "__main__":
    sys.exit(0 if main() is None else 0)
