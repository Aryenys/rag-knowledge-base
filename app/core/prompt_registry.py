"""
Prompt 版本管理（任务4）。

解决三个实际痛点：
  1. 改了 prompt，线上答案变差，却不知道是哪个版本引起的——改完就覆盖，无从回溯
  2. 想试新 prompt 只能直接改代码重启，没法灰度、没法快速回滚
  3. 改了到底变好还是变差，只能凭感觉，没有对比手段

做法（刻意保持简单，不引入数据库/配置中心）：
  - 每个版本是一个 JSON 文件，放在本文件同级的 prompt_versions/ 下
  - 版本之间用 extends 继承：v2 只写「跟 v1 不一样的那一个模板」，其余自动继承
  - 默认版本由环境变量 PROMPT_VERSION 决定，取不到就回退 v1，不让服务起不来

为什么用文件而不是数据库：
  prompt 是跟代码一起评审、一起发布的东西，放进 Git 才能 diff、才能 code review。
  放数据库反而脱离了版本控制——那是把资产放进了最难审计的地方。
"""
import json
import logging
import re
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_VERSION_DIR = Path(__file__).parent / "prompt_versions"
_FALLBACK_VERSION = "v1"

# 模板里允许出现的变量。加载时就校验，写错占位符立刻报错，
# 而不是等到线上某次问答才在 .format() 那里炸掉。
_ALLOWED_VARS = {"context", "history", "question", "index", "filename", "page_info", "content"}
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

_cache: dict[str, dict] = {}
_meta_cache: list[dict] | None = None


class PromptVersionError(Exception):
    """版本文件缺失、继承链成环、模板含未知占位符时抛出。"""


def list_versions() -> list[dict]:
    """列出所有可用版本的元信息（不加载模板内容，很轻）。"""
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache

    metas = []
    for path in sorted(_VERSION_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[prompt] 版本文件解析失败，已跳过：%s（%s）", path.name, e)
            continue
        metas.append({
            "version": data.get("version", path.stem),
            "name": data.get("name", ""),
            "created_at": data.get("created_at", ""),
            "extends": data.get("extends"),
            "notes": data.get("notes", ""),
            "changes": data.get("changes", []),
            "templates": sorted(data.get("templates", {}).keys()),
        })
    _meta_cache = metas
    return metas


def load_version(version: str) -> dict:
    """加载某个版本的最终模板集合（解析 extends 继承链后合并）。

    继承链从最基础的祖先开始往上覆盖，所以子版本只写差异即可。
    """
    if version in _cache:
        return _cache[version]

    chain: list[dict] = []
    seen: set[str] = set()
    cursor: str | None = version

    while cursor:
        if cursor in seen:
            raise PromptVersionError(f"版本继承链成环：{cursor}")
        seen.add(cursor)

        path = _VERSION_DIR / f"{cursor}.json"
        if not path.exists():
            raise PromptVersionError(f"版本不存在：{cursor}（文件 {path} 缺失）")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise PromptVersionError(f"版本文件不是合法 JSON：{path.name}（{e}）") from e

        chain.append(data)
        cursor = data.get("extends")

    merged: dict[str, str] = {}
    for data in reversed(chain):          # 从祖先往子孙覆盖
        merged.update(data.get("templates", {}))

    _validate_templates(merged, version)
    result = {"version": version, "templates": merged, "meta": chain[0]}
    _cache[version] = result
    return result


def _validate_templates(templates: dict[str, str], version: str) -> None:
    """校验占位符合法性：未知变量一律拒绝。

    这一步的价值在于「fail fast」——prompt 写错在运行前就暴露，
    而不是等某条问答触发到那个模板才崩。
    """
    for name, text in templates.items():
        for var in _PLACEHOLDER_RE.findall(text):
            if var not in _ALLOWED_VARS:
                raise PromptVersionError(
                    f"版本 {version} 的模板 {name} 含未知占位符 {{{var}}}，"
                    f"允许的变量：{sorted(_ALLOWED_VARS)}"
                )


def get_template(name: str, version: str | None = None) -> str:
    """取某个模板。version 为空时用配置里的默认版本。

    版本加载失败（配错版本号、文件被删、JSON 写坏）时回退 v1 并告警：
    prompt 出问题不应该让整个问答服务起不来。
    """
    target = version or settings.PROMPT_VERSION
    try:
        data = load_version(target)
    except PromptVersionError as e:
        logger.warning("[prompt] 加载版本 %s 失败（%s），回退 %s", target, e, _FALLBACK_VERSION)
        data = load_version(_FALLBACK_VERSION)

    if name not in data["templates"]:
        raise KeyError(f"模板 {name} 在版本 {data['version']} 中不存在")
    return data["templates"][name]


def render(name: str, version: str | None = None, **kwargs: object) -> str:
    """渲染模板。

    为什么不用 str.format()：format 遇到模板里任何花括号都会当占位符处理，
    写个 JSON 示例就会 KeyError。这里只替换「确实传进来的变量」，
    其余原样保留，模板里写示例代码也不会炸。
    """
    template = get_template(name, version)

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        return str(kwargs[key]) if key in kwargs else match.group(0)

    return _PLACEHOLDER_RE.sub(_sub, template)


def current_version() -> str:
    """当前实际生效的版本号（回退发生时这里反映回退后的真实版本）。"""
    return _effective_version()


def _effective_version() -> str:
    try:
        return load_version(settings.PROMPT_VERSION)["version"]
    except PromptVersionError:
        return _FALLBACK_VERSION
