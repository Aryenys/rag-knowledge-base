"""W4 Step3 验证脚本：多轮对话（指代消解）+ 消息落库。

用法（项目根目录、venv 已激活）：
    python test_multi_turn.py

验证三件事：
  1. 同一个 session_id 下连续提问，第二轮带"它"能否被正确理解
  2. 查询改写把"它和 302 有什么区别"补全成了什么
  3. 用户提问、助手回答、引用来源是否都写进了 MySQL
"""
import json

from app.services import chat_service, session_service
from app.rag import generator


def consume(question: str, session_id: int):
    """消费 SSE 生成器，拆解各类型帧并返回 (会话id, 完整回答, 引用来源)"""
    answer, sources, got_id = "", [], None
    for frame in chat_service.stream_answer(question, session_id):
        # 帧格式固定为  "data: {json}\n\n"
        payload = json.loads(frame[len("data: "):].strip())
        if payload["type"] == "session":
            got_id = payload["id"]
        elif payload["type"] == "token":
            answer += payload["content"]
        elif payload["type"] == "sources":
            sources = payload["content"]
    return got_id, answer, sources


def show_sources(sources):
    for s in sources:
        print(f"      [{s['index']}] {s['filename']}  分数={s['score']}")
        print(f"          {s['content'][:60]}...")


def main():
    print("=" * 66)
    print("Step3 多轮对话验证")
    print("=" * 66)

    session = session_service.create_session()
    sid = session["id"]
    print(f"\n新建会话 id={sid}\n")

    # ---------- 第一轮 ----------
    print("【第一轮】304 状态码是什么意思")
    print("-" * 66)
    _, ans1, src1 = consume("304 状态码是什么意思", sid)
    print(f"\n回答: {ans1}")
    print(f"引用 {len(src1)} 条:")
    show_sources(src1)

    # ---------- 展示改写效果 ----------
    # 从数据库取出刚落库的历史，看看带"它"的问题会被模型补全成什么
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in session_service.get_recent_messages(sid, 10)
    ]
    print("\n" + "=" * 66)
    print("指代消解演示（这一步就是查询改写在做的事）")
    print("-" * 66)
    q2 = "它和 302 有什么区别"
    rewritten = generator.rewrite_query(q2, history)
    print(f"  用户原话      : {q2}")
    print(f"  改写后用于检索: {rewritten}")
    print("  ↑ '它' 被替换成了具体的主语，检索器这下能看懂了")

    # ---------- 第二轮 ----------
    print("\n" + "=" * 66)
    print("【第二轮】它和 302 有什么区别")
    print("-" * 66)
    _, ans2, src2 = consume(q2, sid)
    print(f"\n回答: {ans2}")
    print(f"引用 {len(src2)} 条:")
    show_sources(src2)

    # ---------- 落库验证 ----------
    print("\n" + "=" * 66)
    print("数据库落库检查")
    print("-" * 66)
    print(f"会话标题: {session_service.get_session(sid)['title']}")
    print(f"（应为第一轮问题的前 20 字，说明标题自动更新生效）\n")
    for m in session_service.get_messages(sid):
        src_cnt = len(m["sources"]) if m["sources"] else 0
        print(f"  [id={m['id']}] {m['role']:<9} 引用{src_cnt}条  {m['content'][:40]}")
    print(f"\n会话 id={sid} 已保留，可在 Swagger 的 GET /api/chat/sessions/{sid}/messages 查看")


if __name__ == "__main__":
    main()
