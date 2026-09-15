"""
【W4·Step2】会话与消息的数据访问层（DAO / Repository）。

职责：只做数据库增删改查，不含业务逻辑。Session 的开关、事务边界全在这里，
上层 service 拿到的是干净的 dict，不接触 ORM 对象。

━━━━━━ 设计决策：为什么不用 Depends(get_db) 注入 Session？ ━━━━━━

方案 A（本项目采用）：每个函数自己 SessionLocal() 开、finally 里关
方案 B：FastAPI 依赖注入 db: Session = Depends(get_db)，再传给 service

选 A 的三个理由：
  1. service 层不依赖 FastAPI —— 命令行脚本、定时任务能直接调用这些函数。
     我们前面用命令行调试 ingester/reranker 就是这个思路的延续。
  2. SSE 流式接口的生成器要跑几十秒，把 db session 的生命周期绑到单个
     HTTP 请求上并不合适（连接长时间占用）；按需开、用完即关更稳。
  3. 事务边界清晰：每个函数自己 commit / rollback，出错不会影响其他操作。
代价：一个请求里调多次会产生多个短连接（靠连接池复用，开销很小）。
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.time_utils import ensure_utc, utc_now
from app.models.db_models import ChatMessage, ChatSession

TITLE_MAX_LEN = 20  # 会话标题取首条问题的前 N 字


# ---------- ORM 对象 -> dict（必须在 session 关闭前完成） ----------

def _session_to_dict(s: ChatSession) -> dict:
    """⚠️ 注意：ORM 对象一旦脱离 session 就是 DetachedInstanceError，必须先转成 dict"""
    return {"id": s.id, "title": s.title, "created_at": ensure_utc(s.created_at)}


def _message_to_dict(m: ChatMessage) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "sources": m.sources,  # JSON 列：None（user）或引用来源列表（assistant）
        "created_at": ensure_utc(m.created_at),
    }


# ---------- 会话 ----------

def create_session(title: str = "新会话") -> dict:
    """新建会话，返回带自增 id 的会话信息"""
    db: Session = SessionLocal()
    try:
        session = ChatSession(title=title)
        db.add(session)
        db.commit()
        db.refresh(session)  # commit 后回读数据库，拿到自增 id 和 created_at 默认值
        return _session_to_dict(session)
    except Exception:
        db.rollback()  # 写操作失败必须回滚，否则事务持有锁不放
        raise
    finally:
        db.close()


def get_session(session_id: int) -> dict | None:
    """按 id 查会话，不存在返回 None（由 API 层决定是否转 404）"""
    db: Session = SessionLocal()
    try:
        session = db.get(ChatSession, session_id)  # 主键查询，比 select 更简洁
        return _session_to_dict(session) if session else None
    finally:
        db.close()


def list_sessions() -> list[dict]:
    """会话列表，按创建时间倒序（最新会话显示在侧边栏最上方）"""
    db: Session = SessionLocal()
    try:
        stmt = select(ChatSession).order_by(ChatSession.id.desc())
        return [_session_to_dict(s) for s in db.execute(stmt).scalars().all()]
    finally:
        db.close()


def update_session_title(session_id: int, title: str) -> None:
    """更新会话标题（首条回答后把'新会话'换成用户第一个问题的摘要）"""
    db: Session = SessionLocal()
    try:
        session = db.get(ChatSession, session_id)
        if session:
            session.title = title[:TITLE_MAX_LEN]
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_session(session_id: int) -> bool:
    """删除会话。关联消息由 relationship 的 cascade 自动级联删除，不用手动删。返回是否找到"""
    db: Session = SessionLocal()
    try:
        session = db.get(ChatSession, session_id)
        if session is None:
            return False
        db.delete(session)  # ORM 会加载 messages 并按 cascade="all, delete-orphan" 一并删除
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------- 消息 ----------

def save_message(session_id: int, role: str, content: str, sources: list | None = None) -> dict:
    """落库一条消息。role=user 或 assistant；sources 仅 assistant 携带引用来源"""
    db: Session = SessionLocal()
    try:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            created_at=utc_now(),
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return _message_to_dict(msg)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_messages(session_id: int, limit: int = 100) -> list[dict]:
    """某会话的完整消息列表，按时间正序（前端渲染历史用）"""
    db: Session = SessionLocal()
    try:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
            .limit(limit)
        )
        return [_message_to_dict(m) for m in db.execute(stmt).scalars().all()]
    finally:
        db.close()


def get_recent_messages(session_id: int, limit: int = 10) -> list[dict]:
    """取最近 N 条消息，返回时仍按时间正序（喂给 LLM 的顺序必须是先远后近）。

    实现技巧：先按 id 倒序取最近 N 条，再反转成正序。
    直接正序 + limit 会拿到最早的 N 条（错），必须先在数据库侧倒序截断。
    Step3 的 chat_service 就用这个函数加载多轮历史。
    """
    db: Session = SessionLocal()
    try:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
        rows = db.execute(stmt).scalars().all()
        return list(reversed([_message_to_dict(m) for m in rows]))
    finally:
        db.close()


if __name__ == "__main__":
    # 自检：跑一遍完整 CRUD 生命周期，用临时数据验证，最后清理干净
    print("=" * 60)
    print("Step2 数据访问层自检")
    print("=" * 60)

    print("\n[1] 新建会话")
    s = create_session("测试会话")
    print(f"    {s}")

    print("\n[2] 查单个会话")
    print(f"    {get_session(s['id'])}")
    print(f"    查不存在的 id=99999 -> {get_session(99999)}")

    print("\n[3] 写入 3 条消息")
    save_message(s["id"], "user", "厦航奖学金的奖励标准是什么")
    save_message(
        s["id"],
        "assistant",
        "金额为每人5000元。",
        sources=[{"index": 1, "filename": "test.md", "page": None, "content": "片段", "score": 0.9}],
    )
    save_message(s["id"], "user", "它什么时候截止申请")
    print(f"    get_messages -> {len(get_messages(s['id']))} 条")
    for m in get_messages(s["id"]):
        src = "有" if m["sources"] else "无"
        print(f"      [{m['role']}] {m['content'][:20]}  引用:{src}")

    print("\n[4] get_recent_messages(limit=2) 应保持正序且取最后两条")
    for m in get_recent_messages(s["id"], limit=2):
        print(f"      [{m['role']}] {m['content']}")

    print("\n[5] 会话列表")
    for item in list_sessions():
        print(f"      {item}")

    print("\n[6] 删除会话（级联删消息）")
    print(f"    delete_session -> {delete_session(s['id'])}")
    print(f"    删除后 get_messages -> {len(get_messages(s['id']))} 条（应为 0）")

    print("\n自检完成")
