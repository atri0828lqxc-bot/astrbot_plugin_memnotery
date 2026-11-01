import asyncio
import functools
import json
import os
import shutil
import sqlite3
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart
from astrbot.api.star import Context, Star, register
from astrbot.api.web import request
from astrbot.core.config import AstrBotConfig
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
from quart import request as quart_request


def _locked(method):
    """串行化 store 访问：配合 check_same_thread=False 与 to_thread 使用。"""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


def _safe_json_tags(raw: str, row_id: Any = None) -> list:
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except Exception:
        logger.warning(f"Memnotery: broken tags JSON at row {row_id}, fallback to []")
        return []


def _clean_iso_time(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    local_tz = datetime.now().astimezone().tzinfo
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=local_tz)
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=local_tz).astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    logger.warning(f"Memnotery: invalid time value dropped: {value!r}")
    return None


AI_IMPORT_SYSTEM = (
    "你是记忆条目解析器。把用户提供的文本解析为 JSON 数组，每个元素固定格式：\n"
    '{"content": "记忆原文", "star": 0到5的整数, "tags": ["标签1", "标签2"], "time": "YYYY-MM-DD HH:MM:SS 或 null"}\n'
    "规则：\n"
    "1. 严禁更改、缩写、翻译、润色或合并 content，必须原样保留文本中每一条记忆的内容\n"
    "2. time 根据文本中的时间信息判断；无法判断时用 null\n"
    "3. star 表示重要程度：0 不重要，3 一般，5 极其重要\n"
    "4. tags 从内容中提炼 1-3 个关键词\n"
    "5. 只输出 JSON 数组本身，不要输出任何解释、markdown 代码块或其他内容"
)


def _relaxed_json(text: str) -> Optional[Any]:
    """宽松解析：兼容尾随逗号、全角引号定界与缺失的数组闭合（旧记忆插件导出文件）。"""
    fixed = re.sub(r",\s*([\]}])", r"\1", text)
    fixed = re.sub(r"(:\s*)\u201c", '\\1"', fixed)
    fixed = re.sub(r"\u201d(\s*[,\}\]])", '"\\1', fixed)
    for candidate in (text, fixed):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    stripped = fixed.rstrip()
    if stripped.endswith("}") and stripped.count("[") > stripped.count("]"):
        idx = stripped.rfind("}")
        try:
            return json.loads(stripped[:idx] + "]" + stripped[idx:])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


class MemnoteryStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir = Path(path).parent / "backups"
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                original_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                context TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                star INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL DEFAULT 'long',
                deleted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope, deleted);
            CREATE TABLE IF NOT EXISTS shared_scopes (
                scope TEXT PRIMARY KEY,
                target_scope TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_scopes (
                scope TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_aliases (
                scope TEXT PRIMARY KEY,
                alias TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """)
        self.db.commit()
        self._known_scopes: set[str] = {row[0] for row in self.db.execute("SELECT scope FROM session_scopes")}
        self._migrate_legacy_numbers()

    def _migrate_legacy_numbers(self) -> None:
        """一次性把存量序号重排为每会话独立编号：活跃行 1..n，垃圾箱行续号 n+1..m。
        迁移后运行期维持会话内永不复用（add 取该会话 MAX+1，删除不回收号），保证序号稳定可引用；
        settings 表标记 per_scope_numbering 防止每次启动重复重排导致序号漂移。"""
        try:
            if self.get_settings().get("per_scope_numbering") == "1":
                return
            with self._lock:
                for scope_row in self.db.execute("SELECT DISTINCT scope FROM memories").fetchall():
                    self._renumber_scope(scope_row[0])
                self.set_settings({"per_scope_numbering": "1"})
                self.db.commit()
                logger.info("Memnotery: renumbered memories per-scope (one-time migration)")
        except Exception:
            logger.exception("Memnotery per-scope renumber migration failed")

    @_locked
    def _backup(self, reason: str = "manual") -> Optional[str]:
        """破坏性操作前快照整个库，保留最近 5 份。"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = self.backup_dir / f"memnotery-{stamp}-{reason}.db"
            target = sqlite3.connect(str(dest))
            try:
                self.db.backup(target)
                target.commit()
            finally:
                target.close()  # 必须显式关闭，否则 Windows 上文件句柄占用导致备份无法清理
            backups = sorted(self.backup_dir.glob("memnotery-*.db"))
            for old in backups[:-5]:
                old.unlink(missing_ok=True)
            return str(dest)
        except Exception:
            logger.exception("Memnotery backup failed")
            return None

    def close(self) -> None:
        """关闭数据库连接，释放文件句柄（供插件 terminate 时调用）。"""
        try:
            self.db.close()
        except Exception:
            pass

    @_locked
    def register_scope(self, scope: str) -> None:
        if scope in self._known_scopes:
            return
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT INTO session_scopes(scope, created_at, last_seen_at) VALUES(?,?,?) "
            "ON CONFLICT(scope) DO UPDATE SET last_seen_at=excluded.last_seen_at",
            (scope, now, now),
        )
        self.db.commit()
        self._known_scopes.add(scope)

    def prune_scopes(self) -> int:
        with self._lock:
            cur = self.db.execute(
                "DELETE FROM session_scopes WHERE scope NOT IN ("
                "SELECT DISTINCT scope FROM memories UNION "
                "SELECT scope FROM shared_scopes UNION SELECT target_scope FROM shared_scopes UNION "
                "SELECT scope FROM session_aliases)"
            )
            self.db.commit()
            if cur.rowcount:
                self._known_scopes = {row[0] for row in self.db.execute("SELECT scope FROM session_scopes")}
            return cur.rowcount

    @_locked
    def get_aliases(self) -> dict[str, str]:
        return {row[0]: row[1] for row in self.db.execute("SELECT scope, alias FROM session_aliases")}

    @_locked
    def set_alias(self, scope: str, alias: str) -> None:
        self.db.execute(
            "INSERT INTO session_aliases(scope, alias) VALUES(?,?) "
            "ON CONFLICT(scope) DO UPDATE SET alias=excluded.alias",
            (scope, alias.strip()[:50]),
        )
        self.db.commit()

    @_locked
    def get_settings(self) -> dict[str, str]:
        return {row[0]: row[1] for row in self.db.execute("SELECT key, value FROM settings")}

    @_locked
    def set_settings(self, values: dict[str, str]) -> None:
        for key, value in values.items():
            self.db.execute(
                "INSERT INTO settings(key, value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
        self.db.commit()

    @_locked
    def create_session(self, scope: str, alias: str) -> str:
        self.register_scope(scope)
        self.set_alias(scope, alias)
        return scope

    @_locked
    def delete_session(self, scope: str) -> tuple[bool, str]:
        """删除会话条目。仅允许删除没有任何记忆（含垃圾箱）的会话；组根一并解除映射。"""
        count = self.db.execute("SELECT COUNT(*) FROM memories WHERE scope=?", (scope,)).fetchone()[0]
        if count:
            return False, f"会话内还有 {count} 条记忆（含垃圾箱），请先删除记忆后再删除会话"
        self.db.execute("DELETE FROM shared_scopes WHERE scope=? OR target_scope=?", (scope, scope))
        self.db.execute("DELETE FROM session_aliases WHERE scope=?", (scope,))
        self.db.execute("DELETE FROM session_scopes WHERE scope=?", (scope,))
        self.db.commit()
        self._known_scopes.discard(scope)
        return True, "ok"

    @_locked
    def list_scopes(self) -> list[str]:
        rows = self.db.execute(
            "SELECT scope FROM session_scopes UNION SELECT DISTINCT scope FROM memories ORDER BY scope"
        ).fetchall()
        return [row[0] for row in rows]

    @_locked
    def resolve_scope(self, scope: str) -> str:
        mapping = {row[0]: row[1] for row in self.db.execute("SELECT scope, target_scope FROM shared_scopes")}
        seen: set[str] = set()
        while scope not in seen and scope in mapping and mapping[scope] != scope:
            seen.add(scope)
            scope = mapping[scope]
        return scope

    @_locked
    def group_scopes(self, scope: str) -> list[str]:
        mapping = {row[0]: row[1] for row in self.db.execute("SELECT scope, target_scope FROM shared_scopes")}

        def resolve(item: str) -> str:
            seen: set[str] = set()
            while item not in seen and item in mapping and mapping[item] != item:
                seen.add(item)
                item = mapping[item]
            return item

        root = resolve(scope)
        scopes = set(self.list_scopes()) | {scope, root}
        return sorted(item for item in scopes if resolve(item) == root)

    @_locked
    def list_tags(self, scope: str) -> list[str]:
        rows = self.db.execute(
            "SELECT id, tags FROM memories WHERE scope=? AND deleted=0 AND kind='long'",
            (self.resolve_scope(scope),),
        ).fetchall()
        tags = {tag for row in rows for tag in _safe_json_tags(row["tags"], row["id"])}
        return sorted(tags)

    @_locked
    def update(self, scope: str, number: int, context: Optional[str] = None,
               tags: Optional[list[str]] = None, star: Optional[int] = None,
               time: Optional[str] = None) -> bool:
        scope = self.resolve_scope(scope)
        row = self.db.execute(
            "SELECT id FROM memories WHERE scope=? AND original_id=? AND deleted=0",
            (scope, number),
        ).fetchone()
        if not row:
            return False
        updates = []
        values = []
        if context is not None:
            updates.append("context=?")
            values.append(context)
        if tags is not None:
            updates.append("tags=?")
            values.append(json.dumps(tags, ensure_ascii=False))
        if star is not None:
            updates.append("star=?")
            values.append(max(0, min(5, int(star))))
        if time is not None:
            updates.append("time=?")
            values.append(str(time))
        if not updates:
            return True
        values.append(row["id"])
        self.db.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id=?", values)
        self.db.commit()
        return True

    @_locked
    def migrate(self, source: str, target: str, row_ids: Optional[list[int]] = None) -> int:
        source_root, target_root = self.resolve_scope(source), self.resolve_scope(target)
        if source_root == target_root:
            return 0
        if row_ids is None:
            rows = self.db.execute(
                "SELECT id FROM memories WHERE scope=? AND deleted=0 ORDER BY original_id",
                (source_root,),
            ).fetchall()
            row_ids = [row[0] for row in rows]
        row_ids = sorted(set(int(row_id) for row_id in row_ids))
        if not row_ids:
            return 0
        placeholders = ",".join("?" for _ in row_ids)
        self._backup("migrate")
        self.db.execute("BEGIN IMMEDIATE")
        rows = self.db.execute(
            f"SELECT id FROM memories WHERE scope=? AND deleted=0 AND id IN ({placeholders}) ORDER BY original_id",
            [source_root, *row_ids],
        ).fetchall()
        if not rows:
            self.db.rollback()
            return 0
        next_id = self.db.execute(
            "SELECT COALESCE(MAX(original_id), 0) FROM memories WHERE scope=?", (target_root,)
        ).fetchone()[0]
        for offset, row in enumerate(rows, 1):
            self.db.execute(
                "UPDATE memories SET scope=?, original_id=? WHERE id=?",
                (target_root, next_id + offset, row["id"]),
            )
        self.db.commit()
        return len(rows)

    def _renumber_scope(self, scope: str) -> None:
        """组内统一重编号：活跃行 1..n，垃圾箱行续号 n+1..m（编号永不复用）。"""
        rows = self.db.execute(
            "SELECT id FROM memories WHERE scope=? ORDER BY deleted, original_id, id", (scope,)
        ).fetchall()
        for index, row in enumerate(rows, 1):
            self.db.execute("UPDATE memories SET original_id=? WHERE id=?", (index, row["id"]))

    def _copy_scope_memories(self, source: str, target: str) -> None:
        rows = self.db.execute("SELECT * FROM memories WHERE scope=? ORDER BY id", (source,)).fetchall()
        for row in rows:
            self.db.execute(
                "INSERT INTO memories(scope, original_id, time, context, tags, star, kind, deleted, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (target, row["original_id"], row["time"], row["context"], row["tags"], row["star"],
                 row["kind"], row["deleted"], row["created_at"]),
            )
        self._renumber_scope(target)

    @_locked
    def share(self, scope: str, target: str) -> int:
        source_root, target_root = self.resolve_scope(scope), self.resolve_scope(target)
        if source_root == target_root:
            return 0
        source_members = self.group_scopes(source_root)
        target_members = self.group_scopes(target_root)
        source_rows = self.db.execute(
            "SELECT id FROM memories WHERE scope=? AND deleted=0 ORDER BY original_id", (source_root,)
        ).fetchall()
        self._backup("share")
        self.db.execute("BEGIN IMMEDIATE")
        self.db.execute("UPDATE memories SET scope=? WHERE scope=?", (target_root, source_root))
        self._renumber_scope(target_root)
        for member in set(source_members + target_members):
            self.db.execute("DELETE FROM shared_scopes WHERE scope=?", (member,))
            if member != target_root:
                self.db.execute(
                    "INSERT INTO shared_scopes(scope,target_scope) VALUES(?,?)", (member, target_root)
                )
        self.db.commit()
        return len(source_rows)

    @_locked
    def unshare(self, scope: str, target: Optional[str] = None) -> int:
        root = self.resolve_scope(scope)
        member = target or scope
        members = self.group_scopes(scope)
        if member not in members or len(members) < 2:
            return 0
        self._backup("unshare")
        self.db.execute("BEGIN IMMEDIATE")
        if member == root:
            remaining = [item for item in members if item != root]
            new_root = remaining[0]
            self.db.execute("UPDATE memories SET scope=? WHERE scope=?", (new_root, root))
            self._renumber_scope(new_root)
            self._copy_scope_memories(new_root, root)
            self._renumber_scope(root)
            for item in remaining:
                self.db.execute("DELETE FROM shared_scopes WHERE scope=?", (item,))
                if item != new_root:
                    self.db.execute("INSERT INTO shared_scopes(scope,target_scope) VALUES(?,?)", (item, new_root))
            self.db.execute("DELETE FROM shared_scopes WHERE scope=?", (root,))
        else:
            self._copy_scope_memories(root, member)
            self.db.execute("DELETE FROM shared_scopes WHERE scope=?", (member,))
        self.db.commit()
        return 1

    @_locked
    def add(self, scope: str, context: str, tags: list[str], star: int = 0,
            kind: str = "long", time: Optional[str] = None) -> dict[str, Any]:
        scope = self.resolve_scope(scope)
        now = _clean_iso_time(time) or datetime.now(timezone.utc).isoformat()
        self.db.execute("BEGIN IMMEDIATE")
        row = self.db.execute(
            "SELECT COALESCE(MAX(original_id), 0) + 1 FROM memories WHERE scope=?", (scope,)).fetchone()
        number = int(row[0])
        cur = self.db.execute(
            "INSERT INTO memories(scope, original_id, time, context, tags, star, kind, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (scope, number, now, context, json.dumps(tags, ensure_ascii=False), max(0, min(5, int(star))), kind, now))
        self.db.commit()
        row = self.db.execute("SELECT * FROM memories WHERE id=?", (cur.lastrowid,)).fetchone()
        return self._format(row)

    @_locked
    def get(self, scope: str, row_id: int) -> Optional[dict[str, Any]]:
        row = self.db.execute("SELECT * FROM memories WHERE scope=? AND id=?", (self.resolve_scope(scope), row_id)).fetchone()
        return self._format(row) if row else None

    @_locked
    def search(self, scope: str, query: str = "", tags: Optional[list[str]] = None, kind: Optional[str] = None,
               include_deleted: bool = False, only_deleted: bool = False, limit: int = 20,
               include_key: bool = False) -> list[dict[str, Any]]:
        scope = self.resolve_scope(scope)
        clauses = ["scope=?"]
        args: list[Any] = [scope]
        if only_deleted:
            clauses.append("deleted=1")
        elif not include_deleted:
            clauses.append("deleted=0")
        if kind in ("prompt", "long"):
            clauses.append("kind=?")
            args.append(kind)
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            needle = f"%{escaped}%"
            clauses.append("(context LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')")
            args.extend([needle, needle])
        sql = "SELECT * FROM memories WHERE " + " AND ".join(clauses) + " ORDER BY original_id LIMIT ?"
        args.append(max(1, min(100, limit)))
        rows = self.db.execute(sql, args).fetchall()
        result = [dict(self._format(row), row_id=row["id"]) if include_key else self._format(row) for row in rows]
        if tags:
            result = [item for item in result if set(tags).intersection(item["tags"])]
        return result

    def remove(self, scope: str, number: int) -> bool:
        return self.remove_many(scope, [number]) > 0

    @_locked
    def remove_many(self, scope: str, numbers: list[int]) -> int:
        scope = self.resolve_scope(scope)
        numbers = sorted(set(int(number) for number in numbers))
        if not numbers:
            return 0
        placeholders = ",".join("?" for _ in numbers)
        rows = self.db.execute(
            f"SELECT id FROM memories WHERE scope=? AND deleted=0 AND original_id IN ({placeholders})",
            [scope, *numbers],
        ).fetchall()
        return self.remove_rows(scope, [row["id"] for row in rows])

    @_locked
    def remove_rows(self, scope: str, row_ids: list[int]) -> int:
        scope = self.resolve_scope(scope)
        row_ids = sorted(set(int(row_id) for row_id in row_ids))
        if not row_ids:
            return 0
        placeholders = ",".join("?" for _ in row_ids)
        self.db.execute("BEGIN IMMEDIATE")
        cur = self.db.execute(
            f"UPDATE memories SET deleted=1 WHERE scope=? AND deleted=0 AND id IN ({placeholders})",
            [scope, *row_ids],
        )
        self.db.commit()
        return cur.rowcount

    @_locked
    def restore(self, scope: str, number: int, row_id: Optional[int] = None) -> bool:
        scope = self.resolve_scope(scope)
        self.db.execute("BEGIN IMMEDIATE")
        if row_id is None:
            row = self.db.execute("SELECT * FROM memories WHERE scope=? AND original_id=? AND deleted=1 ORDER BY id DESC LIMIT 1", (scope, number)).fetchone()
        else:
            row = self.db.execute("SELECT * FROM memories WHERE scope=? AND id=? AND deleted=1", (scope, row_id)).fetchone()
        if not row:
            self.db.rollback()
            return False
        self.db.execute("UPDATE memories SET deleted=0 WHERE id=?", (row["id"],))
        self.db.commit()
        return True

    @_locked
    def restore_many(self, scope: str, row_ids: list[int]) -> int:
        scope = self.resolve_scope(scope)
        restored = 0
        for row_id in sorted(set(int(value) for value in row_ids)):
            row = self.db.execute("SELECT original_id FROM memories WHERE scope=? AND id=? AND deleted=1", (scope, row_id)).fetchone()
            if row and self.restore(scope, row["original_id"], row_id):
                restored += 1
        return restored

    @_locked
    def purge(self, scope: str) -> int:
        scope = self.resolve_scope(scope)
        count = self.db.execute("SELECT COUNT(*) FROM memories WHERE scope=? AND deleted=1", (scope,)).fetchone()[0]
        if count:
            self._backup("purge")
        cur = self.db.execute("DELETE FROM memories WHERE scope=? AND deleted=1", (scope,))
        self.db.commit()
        return cur.rowcount

    @_locked
    def purge_rows(self, scope: str, row_ids: list[int]) -> int:
        row_ids = sorted(set(int(row_id) for row_id in row_ids))
        if not row_ids:
            return 0
        placeholders = ",".join("?" for _ in row_ids)
        if self.db.execute(
            f"SELECT COUNT(*) FROM memories WHERE scope=? AND deleted=1 AND id IN ({placeholders})",
            [self.resolve_scope(scope), *row_ids],
        ).fetchone()[0]:
            self._backup("purge")
        cur = self.db.execute(
            f"DELETE FROM memories WHERE scope=? AND deleted=1 AND id IN ({placeholders})",
            [self.resolve_scope(scope), *row_ids],
        )
        self.db.commit()
        return cur.rowcount

    @_locked
    def export_data(self, scope: str, include_deleted: bool = True) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM memories WHERE scope=? AND (? OR deleted=0) ORDER BY original_id",
            (self.resolve_scope(scope), int(include_deleted)),
        ).fetchall()
        return [self._format(row) for row in rows]

    @_locked
    def import_data(self, scope: str, items: Any) -> dict[str, int]:
        if isinstance(items, dict):
            if isinstance(items.get("items"), list):
                items = items["items"]
            else:
                grouped = []
                for group, kind in (("core_memory", "prompt"), ("medium_term", "prompt"), ("long_term", "long")):
                    entries = items.get(group, {})
                    if isinstance(entries, dict):
                        grouped.extend((item, kind) for item in entries.get("upsert", []) if isinstance(item, dict))
                if grouped:
                    items = grouped
                else:
                    # 旧插件导出格式：{会话: [条目...]}，条目字段 content/importance/timestamp/memory_id
                    flat = []
                    for key, entries in items.items():
                        if not isinstance(entries, list):
                            continue
                        for entry in entries:
                            if isinstance(entry, dict):
                                entry = dict(entry)
                                entry.setdefault("_scope", str(key))
                                flat.append((entry, "prompt" if entry.get("kind") == "prompt" else "long"))
                            elif isinstance(entry, str):
                                flat.append(({"_scope": str(key), "content": entry}, "long"))
                    items = flat
        else:
            items = [(item, "prompt" if isinstance(item, dict) and item.get("kind") == "prompt" else "long")
                     for item in items] if isinstance(items, list) else []
        imported = failed = duplicates = 0
        errors: list[str] = []
        fallback_scope = self.resolve_scope(scope)
        for item, kind in items:
            try:
                item_scope = str(item.get("_scope", "")).strip() or fallback_scope
                context = str(item.get("context", item.get("content", ""))).strip()
                if not context:
                    continue
                raw_tags = item.get("tags", item.get("tag", item.get("category", [])))
                tags = raw_tags if isinstance(raw_tags, list) else [x.strip() for x in str(raw_tags).split(",") if x.strip()]
                star = item.get("star", item.get("importance", 0)) or 0
                time = _clean_iso_time(item.get("time") or item.get("timestamp") or item.get("expires_at"))
                existing = self.db.execute(
                    "SELECT 1 FROM memories WHERE scope=? AND context=? AND kind=? AND deleted=0 LIMIT 1",
                    (item_scope, context, kind)).fetchone()
                if existing:
                    duplicates += 1
                    continue
                self.add(item_scope, context, tags, int(star), kind, time)
                imported += 1
            except Exception as exc:
                failed += 1
                errors.append(str(exc))
                logger.warning(f"Memnotery import item failed: {exc}")
        return {"imported": imported, "failed": failed, "duplicates": duplicates, "errors": errors[:5]}

    def _format(self, row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["original_id"], "time": row["time"], "context": row["context"],
                "tags": _safe_json_tags(row["tags"], row["id"]), "star": row["star"],
                "kind": row["kind"], "deleted": bool(row["deleted"])}


@register("astrbot_plugin_memory", "GLM & GPT-5.4", "Memnotery - 会话记忆、提示记忆与垃圾箱管理插件", "0.2.0")
class MemnoteryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config or {}
        data_dir = self._resolve_data_dir()
        self.store = MemnoteryStore(os.path.join(data_dir, "memory.sqlite3"))
        self._ai_progress: dict | None = None
        self._inject_provider_options()
        context.register_web_api("/astrbot_plugin_memory/api/list", self.api_list, ["GET"], "List memories")
        context.register_web_api("/astrbot_plugin_memory/api/update", self.api_update, ["POST"], "Update memory")
        context.register_web_api("/astrbot_plugin_memory/api/delete", self.api_delete, ["POST"], "Delete memory")
        context.register_web_api("/astrbot_plugin_memory/api/purge-selected", self.api_purge_selected, ["POST"], "Purge selected trash")
        context.register_web_api("/astrbot_plugin_memory/api/migrate-selected", self.api_migrate_selected, ["POST"], "Move selected memories")
        context.register_web_api("/astrbot_plugin_memory/api/restore", self.api_restore, ["POST"], "Restore memory")
        context.register_web_api("/astrbot_plugin_memory/api/purge", self.api_purge, ["POST"], "Purge trash")
        context.register_web_api("/astrbot_plugin_memory/api/add", self.api_add, ["POST"], "Add memory")
        context.register_web_api("/astrbot_plugin_memory/api/import", self.api_import, ["POST"], "Import memories")
        context.register_web_api("/astrbot_plugin_memory/api/import_ai", self.api_import_ai, ["POST"], "AI semantic import")
        context.register_web_api("/astrbot_plugin_memory/api/ai_settings", self.api_ai_settings, ["GET"], "AI import settings")
        context.register_web_api("/astrbot_plugin_memory/api/ai_settings/save", self.api_ai_settings_save, ["POST"], "Save AI import settings")
        context.register_web_api("/astrbot_plugin_memory/api/ai_import_progress", self.api_ai_import_progress, ["GET"], "AI import progress")
        context.register_web_api("/astrbot_plugin_memory/api/export", self.api_export, ["GET"], "Export memories")
        context.register_web_api("/astrbot_plugin_memory/api/sessions", self.api_sessions, ["GET"], "List session scopes")
        context.register_web_api("/astrbot_plugin_memory/api/session/create", self.api_session_create, ["POST"], "Create custom session")
        context.register_web_api("/astrbot_plugin_memory/api/session/rename", self.api_session_rename, ["POST"], "Rename session")
        context.register_web_api("/astrbot_plugin_memory/api/session/delete", self.api_session_delete, ["POST"], "Delete empty session")
        context.register_web_api("/astrbot_plugin_memory/api/share", self.api_share, ["POST"], "Share session memories")
        context.register_web_api("/astrbot_plugin_memory/api/unshare", self.api_unshare, ["POST"], "Stop sharing session memories")
        context.register_web_api("/astrbot_plugin_memory/api/migrate", self.api_migrate, ["POST"], "Move session memories")

    async def terminate(self):
        """AstrBot 插件卸载/重载钩子：关闭数据库连接，避免 Windows 上文件句柄泄漏。"""
        self.store.close()

    def _resolve_data_dir(self) -> str:
        """持久化数据统一存放于 data/plugin_data/astrbot_plugin_memory（审核规范）。
        旧版本把数据库写在插件自身目录 data/ 下，这里一次性整体搬迁（含备份与 WAL）。"""
        new_dir = os.path.join(get_astrbot_plugin_data_path(), "astrbot_plugin_memory")
        if os.path.exists(os.path.join(new_dir, "memory.sqlite3")):
            return new_dir
        os.makedirs(new_dir, exist_ok=True)
        old_dir = os.path.join(os.path.dirname(__file__), "data")
        old_db = os.path.join(old_dir, "memory.sqlite3")
        if not os.path.exists(old_db):
            return new_dir
        try:
            conn = sqlite3.connect(old_db)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
        except Exception:
            logger.exception("Memnotery wal checkpoint before migration failed")
        for name in os.listdir(old_dir):
            src = os.path.join(old_dir, name)
            try:
                if name == "memnotery.db" and os.path.isfile(src):
                    os.remove(src)  # 旧版遗留空库，直接清理
                    continue
                shutil.move(src, os.path.join(new_dir, name))
            except Exception:
                logger.exception(f"Memnotery data migration failed for: {name}")
        try:
            os.rmdir(old_dir)
            logger.info("Memnotery: migrated data dir to " + new_dir)
        except OSError:
            pass
        return new_dir

    def _inject_provider_options(self) -> None:
        """把当前供应商列表动态注入配置页 schema，让 ai_provider_id 在配置页渲染为下拉框。
        热重载后 __init__ 会重新执行，下拉始终跟随最新供应商列表。"""
        try:
            schema = getattr(self.config, "schema", None)
            if not isinstance(schema, dict):
                return
            item = schema.get("ai_provider_id")
            if not isinstance(item, dict):
                return
            options, labels = [""], ["跟随系统当前供应商"]
            for prov in self.context.get_all_providers():
                cfg = getattr(prov, "provider_config", None) or {}
                pid = str(cfg.get("id", "")).strip()
                if not pid:
                    continue
                model = str(cfg.get("model", "") or "")
                options.append(pid)
                labels.append(f"{pid}（{model}）" if model else pid)
                if not cfg.get("enable", False):
                    labels[-1] += "【未启用】"
            item["options"] = options
            item["labels"] = labels
        except Exception:
            logger.exception("Memnotery inject provider options failed")

    def _cfg(self, key: str) -> bool:
        try:
            return bool(self.config.get(key, True))
        except Exception:
            return True

    def _scope(self, event: AstrMessageEvent) -> str:
        scope = str(getattr(event, "unified_msg_origin", None) or "unknown")
        self.store.register_scope(scope)
        return scope

    @filter.llm_tool(name="memory_save")
    async def memory_save(self, event: AstrMessageEvent, memory_text: str, tags: str, star: int = 0,
                          kind: str = "long") -> str:
        """保存一条已经总结好的用户记忆。
        Args:
            memory_text(string): 记忆正文，使用简洁事实描述，不超过 2000 字。
            tags(string): 逗号分隔的标签。
            star(number): 重要程度，0 到 5。
            kind(string): 记忆类型，prompt 表示每轮提示记忆，long 表示按需检索的长期记忆。
        """
        text = str(memory_text).strip()
        if not text:
            return json.dumps({"ok": False, "error": "memory_text is empty"}, ensure_ascii=False)
        if len(text) > 2000:
            return json.dumps({"ok": False, "error": "记忆过长（超过 2000 字），请精简后再保存"}, ensure_ascii=False)
        if kind == "prompt" and not self._cfg("enable_prompt_memory"):
            return json.dumps({"ok": False, "error": "提示记忆功能已停用，未保存"}, ensure_ascii=False)
        if kind != "prompt" and not self._cfg("enable_memory"):
            return json.dumps({"ok": False, "error": "长期记忆功能已停用，未保存"}, ensure_ascii=False)
        tag_list = [item.strip() for item in tags.split(",") if item.strip()]
        item = await asyncio.to_thread(
            self.store.add, self._scope(event), text, tag_list, star,
            "prompt" if kind == "prompt" else "long")
        return json.dumps({"ok": True, "item": item}, ensure_ascii=False)

    @filter.llm_tool(name="memory_update")
    async def memory_update(self, event: AstrMessageEvent, number: int, memory_text: str = "", tags: str = "",
                            star: int = -1) -> str:
        """更新指定序号的长期记忆内容。序号可通过 memory_search 查询获得。
        Args:
            number(number): 要更新的记忆序号。
            memory_text(string): 新的记忆正文，留空表示不修改内容。
            tags(string): 逗号分隔的新标签，留空表示不修改标签。
            star(number): 新的重要程度 0-5，传 -1 表示不修改。
        """
        if not self._cfg("enable_memory"):
            return json.dumps({"ok": False, "error": "长期记忆功能已停用"}, ensure_ascii=False)
        tag_list = [item.strip() for item in tags.split(",") if item.strip()] or None
        context = str(memory_text).strip() or None
        new_star = int(star) if int(star) >= 0 else None
        if context is None and tag_list is None and new_star is None:
            return json.dumps({"ok": False, "error": "没有提供任何要修改的字段"}, ensure_ascii=False)
        ok = await asyncio.to_thread(
            self.store.update, self._scope(event), int(number), context, tag_list, new_star)
        if ok:
            return json.dumps({"ok": True}, ensure_ascii=False)
        return json.dumps({"ok": False, "error": "序号不存在"}, ensure_ascii=False)

    @filter.llm_tool(name="memory_list_tags")
    async def memory_list_tags(self, event: AstrMessageEvent) -> str:
        """列出当前会话长期记忆的标签，供后续按标签检索。注意：仅包含长期记忆，提示记忆每轮自动注入、不可检索。"""
        if not self._cfg("enable_memory"):
            return json.dumps({"ok": False, "error": "长期记忆功能已停用"}, ensure_ascii=False)
        tags = await asyncio.to_thread(self.store.list_tags, self._scope(event))
        return json.dumps({"ok": True, "tags": tags}, ensure_ascii=False)

    @filter.llm_tool(name="memory_search")
    async def memory_search(self, event: AstrMessageEvent, query: str = "", tags: str = "") -> str:
        """检索当前会话长期记忆（仅长期记忆；提示记忆每轮自动注入、不可检索）。可以先获取标签，再按标签与关键词筛选。
        Args:
            query(string): 记忆正文关键词，可以为空。
            tags(string): 逗号分隔的标签，可以为空。
        """
        if not self._cfg("enable_memory"):
            return json.dumps({"ok": False, "error": "长期记忆功能已停用"}, ensure_ascii=False)
        tag_list = [item.strip() for item in tags.split(",") if item.strip()]
        results = await asyncio.to_thread(
            self.store.search, self._scope(event), query, tag_list or None, "long")
        return json.dumps({"ok": True, "items": results}, ensure_ascii=False)

    @filter.llm_tool(name="memory_delete")
    async def memory_delete(self, event: AstrMessageEvent, number: int) -> str:
        """将指定序号的记忆移入垃圾箱，不执行永久删除。
        Args:
            number(number): 要删除的记忆序号。
        """
        if not self._cfg("enable_memory"):
            return json.dumps({"ok": False, "error": "长期记忆功能已停用"}, ensure_ascii=False)
        removed = await asyncio.to_thread(self.store.remove, self._scope(event), int(number))
        if removed:
            return json.dumps({"ok": True}, ensure_ascii=False)
        return json.dumps({"ok": False, "error": "序号不存在"}, ensure_ascii=False)

    async def _request_json(self):
        errors = []
        try:
            data = await request.json(default=None)
            if data is not None:
                return data
        except Exception as exc:
            errors.append(f"AstrBot request JSON: {exc}")
        try:
            return json.loads((await request.body()).decode("utf-8"))
        except Exception as exc:
            errors.append(f"raw request body JSON: {exc}")
        try:
            data = await quart_request.get_json(silent=True)
            if data is not None:
                return data
        except Exception as exc:
            errors.append(f"Quart request JSON: {exc}")
        if errors:
            logger.warning("Memnotery import/request JSON parsing failed: " + "; ".join(errors))
        return {}

    async def api_sessions(self):
        scope = str(request.query.get("scope", ""))
        await asyncio.to_thread(self.store.prune_scopes)
        sessions = await asyncio.to_thread(self.store.list_scopes)
        aliases = await asyncio.to_thread(self.store.get_aliases)
        groups = await asyncio.to_thread(self.store.group_scopes, scope) if scope else []
        return {"sessions": [{"scope": item, "alias": aliases.get(item, "")} for item in sessions],
                "group": groups}

    async def api_session_create(self):
        data = await self._request_json()
        alias = str(data.get("alias", "")).strip()[:50]
        if not alias:
            return {"error": "alias is required"}, 400
        scope = str(data.get("scope", "")).strip() or f"custom-{int(datetime.now().timestamp() * 1000)}"
        if len(scope) > 200 or "?" in scope or "://" in scope:
            return {"error": "invalid scope"}, 400
        await asyncio.to_thread(self.store.create_session, scope, alias)
        return {"scope": scope, "alias": alias}

    async def api_session_rename(self):
        data = await self._request_json()
        scope = str(data.get("scope", "")).strip()
        alias = str(data.get("alias", "")).strip()[:50]
        if not scope or not alias:
            return {"error": "scope and alias are required"}, 400
        await asyncio.to_thread(self.store.set_alias, scope, alias)
        return {"ok": True}

    async def api_session_delete(self):
        data = await self._request_json()
        scope = str(data.get("scope", "")).strip()
        if not scope:
            return {"error": "scope is required"}, 400
        ok, reason = await asyncio.to_thread(self.store.delete_session, scope)
        if not ok:
            return {"error": reason}, 400
        return {"ok": True}

    async def api_list(self):
        scope = str(request.query.get("scope", ""))
        if not scope:
            return {"error": "scope is required"}, 400
        only_deleted = request.query.get("trash", "false").lower() == "true"
        kind = request.query.get("kind")
        return {"items": await asyncio.to_thread(
            self.store.search, scope, request.query.get("q", ""), kind=kind,
            only_deleted=only_deleted, limit=100, include_key=True)}

    async def api_update(self):
        data = await self._request_json()
        if "id" not in data:
            return {"error": "id is required"}, 400
        ok = await asyncio.to_thread(self.store.update, str(data.get("scope", "")), int(data["id"]),
                                     data.get("context"), data.get("tags"), data.get("star"), data.get("time"))
        return {"ok": ok}, (200 if ok else 404)

    async def api_delete(self):
        data = await self._request_json()
        scope = str(data.get("scope", ""))
        if "row_ids" in data:
            return {"deleted": await asyncio.to_thread(self.store.remove_rows, scope, data["row_ids"])}
        if "ids" in data:
            return {"deleted": await asyncio.to_thread(self.store.remove_many, scope, data["ids"])}
        if "id" not in data:
            return {"error": "id is required"}, 400
        ok = await asyncio.to_thread(self.store.remove, scope, int(data["id"]))
        return {"ok": ok}, (200 if ok else 404)

    async def api_purge_selected(self):
        data = await self._request_json()
        deleted = await asyncio.to_thread(
            self.store.purge_rows, str(data.get("scope", "")), data.get("row_ids", []))
        return {"deleted": deleted}

    async def api_migrate_selected(self):
        data = await self._request_json()
        scope, target = str(data.get("scope", "")), str(data.get("target", ""))
        row_ids = data.get("row_ids", [])
        if not scope or not target or scope == target or not isinstance(row_ids, list):
            return {"error": "scope, target and row_ids are required"}, 400
        return {"moved": await asyncio.to_thread(self.store.migrate, scope, target, row_ids)}

    async def api_restore(self):
        data = await self._request_json()
        if "ids" in data:
            return {"restored": await asyncio.to_thread(self.store.restore_many, str(data.get("scope", "")), data["ids"])}
        if "row_id" not in data and "id" not in data:
            return {"error": "id or row_id is required"}, 400
        ok = await asyncio.to_thread(self.store.restore, str(data.get("scope", "")), int(data.get("id", 0)),
                                     int(data["row_id"]) if "row_id" in data else None)
        return {"ok": ok}, (200 if ok else 404)

    async def api_purge(self):
        data = await self._request_json()
        scope = str(data.get("scope", ""))
        if not scope:
            return {"error": "scope is required"}, 400
        return {"deleted": await asyncio.to_thread(self.store.purge, scope)}

    async def api_add(self):
        data = await self._request_json()
        scope = str(data.get("scope", ""))
        context_text = str(data.get("context", "")).strip()
        if not scope or not context_text:
            return {"error": "scope and context are required"}, 400
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [x.strip() for x in tags.split(",") if x.strip()]
        elif not isinstance(tags, list):
            tags = []
        kind = "prompt" if data.get("kind") == "prompt" else "long"
        try:
            star = int(data.get("star", 0) or 0)
        except (TypeError, ValueError):
            star = 0
        item = await asyncio.to_thread(
            self.store.add, scope, context_text, tags, star, kind,
            str(data.get("time")) if data.get("time") else None)
        return {"item": item}

    async def api_import(self):
        try:
            data = await self._request_json()
            body_scope = ""
            if isinstance(data, dict) and isinstance(data.get("raw"), str):
                body_scope = str(data.get("scope") or "")
                parsed = _relaxed_json(data["raw"])
                if parsed is None:
                    return {"error": "文件不是有效的 JSON，宽松修复后仍无法解析，请检查文件内容"}, 400
                data = parsed
            if isinstance(data, list):
                items = data
                scope = body_scope or str(request.query.get("scope", ""))
            elif isinstance(data, dict):
                items = data.get("items")
                scope = str(data.get("scope") or body_scope or request.query.get("scope", ""))
                if items is None and not isinstance(data.get("scope"), str):
                    items = data
            else:
                items = None
                scope = body_scope or str(request.query.get("scope", ""))
            if not scope or not isinstance(items, (list, dict)):
                details = {"body_type": type(data).__name__,
                           "keys": list(data) if isinstance(data, dict) else [],
                           "scope_present": bool(scope), "items_type": type(items).__name__}
                logger.warning(f"Memnotery import rejected: {details}")
                return {"error": "scope and items are required", "received": details}, 400
            result = await asyncio.to_thread(self.store.import_data, scope, items)
            logger.info(f"Memnotery import completed: scope={scope!r}, result={result}")
            return result
        except Exception:
            logger.exception("Memnotery import failed")
            return {"error": "memnotery import failed; see AstrBot backend log"}, 500

    async def api_ai_settings(self):
        providers = []
        try:
            for prov in self.context.get_all_providers():
                cfg = getattr(prov, "provider_config", None) or {}
                providers.append({
                    "id": str(cfg.get("id", "")),
                    "model": str(cfg.get("model", "") or ""),
                    "enable": bool(cfg.get("enable", False)),
                })
        except Exception:
            logger.exception("Memnotery list providers failed")
        settings = await asyncio.to_thread(self.store.get_settings)
        webui_pid = str(settings.get("ai_provider_id") or "").strip()
        cfg_pid = str(self.config.get("ai_provider_id", "") or "").strip()
        return {"providers": providers,
                "provider_id": webui_pid or cfg_pid,
                "provider_source": "webui" if webui_pid else ("config" if cfg_pid else "system"),
                "prompt": settings.get("ai_import_prompt") or str(self.config.get("ai_import_prompt", "") or "")}

    async def api_ai_settings_save(self):
        try:
            data = await self._request_json()
        except Exception:
            return {"error": "invalid JSON body"}, 400
        values = {}
        if "provider_id" in data:
            values["ai_provider_id"] = str(data.get("provider_id") or "").strip()
        if "prompt" in data:
            values["ai_import_prompt"] = str(data.get("prompt") or "").strip()
        if not values:
            return {"error": "nothing to save"}, 400
        await asyncio.to_thread(self.store.set_settings, values)
        return {"ok": True}

    async def api_import_ai(self):
        if not self._cfg("enable_ai_import"):
            return {"error": "AI 智能导入未启用，请先在插件配置中打开 enable_ai_import 并重载插件"}, 403
        try:
            data = await self._request_json()
            scope = str(data.get("scope", "")).strip()
            text = str(data.get("text", "")).strip()
            if not scope:
                return {"error": "scope is required"}, 400
            if not text:
                return {"error": "请提供要解析的文本"}, 400
            if len(text) > 50000:
                return {"error": "文本过长（上限 50000 字符），请分段导入"}, 400
            settings = await asyncio.to_thread(self.store.get_settings)
            pid = (settings.get("ai_provider_id") or str(self.config.get("ai_provider_id", "") or "")).strip()
            provider = None
            if pid:
                try:
                    provider = self.context.get_provider_by_id(pid)
                except Exception:
                    provider = None
                if provider is None:
                    return {"error": f"AI 配置中的供应商 {pid} 不存在或未启用，请在 AI 配置里重新选择"}, 400
            else:
                provider = self.context.get_using_provider()
            if provider is None:
                return {"error": "未配置可用的 LLM 提供商，无法执行 AI 解析"}, 400
            system_prompt = (settings.get("ai_import_prompt") or str(self.config.get("ai_import_prompt", "") or "")).strip() or AI_IMPORT_SYSTEM
            self._ai_progress = {"phase": "parsing", "scope": scope, "total": 0,
                                 "done": 0, "ok": 0, "dup": 0, "failed": 0, "finished": False}
            prompt = f"请把下面的文本解析为记忆条目 JSON 数组：\n\n{text}"
            try:
                response = await provider.text_chat(prompt=prompt, system_prompt=system_prompt)
            except Exception as exc:
                logger.warning(f"Memnotery AI import LLM call failed: {exc}")
                return {"error": f"AI 解析失败：{exc}"}, 500
            raw_output = str(response.completion_text or "")
            parsed = _relaxed_json(raw_output)
            if parsed is None:
                match = re.search(r"\[[\s\S]*\]", raw_output)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except (json.JSONDecodeError, ValueError):
                        parsed = None
            if not isinstance(parsed, list):
                logger.warning(f"Memnotery AI import: LLM output is not a JSON array: {raw_output[:200]}")
                self._ai_progress = {"phase": "idle"}
                return {"error": "AI 返回的内容不是记忆条目数组，已取消导入，请重试或改用直接导入"}, 400
            total = len(parsed)
            self._ai_progress = {"phase": "writing", "scope": scope, "total": total,
                                 "done": 0, "ok": 0, "dup": 0, "failed": 0, "finished": False}
            merged = {"imported": 0, "failed": 0, "duplicates": 0, "errors": []}
            for item in parsed:
                try:
                    r = await asyncio.to_thread(self.store.import_data, scope, [item])
                except Exception as exc:
                    r = {"imported": 0, "failed": 1, "duplicates": 0, "errors": [str(exc)]}
                ok_n = int(r.get("imported", 0) or 0)
                dup_n = int(r.get("duplicates", 0) or 0)
                fail_n = int(r.get("failed", 0) or 0)
                merged["imported"] += ok_n
                merged["duplicates"] += dup_n
                merged["failed"] += fail_n
                errors = r.get("errors")
                if isinstance(errors, list):
                    merged["errors"].extend(str(e) for e in errors[:3])
                progress = dict(self._ai_progress or {})
                progress.update({"done": progress.get("done", 0) + ok_n + dup_n + fail_n,
                                 "ok": progress.get("ok", 0) + ok_n,
                                 "dup": progress.get("dup", 0) + dup_n,
                                 "failed": progress.get("failed", 0) + fail_n})
                self._ai_progress = progress
            self._ai_progress = dict(self._ai_progress, finished=True)
            ai_context = {
                "provider": pid or "（跟随系统当前供应商）",
                "system_prompt": system_prompt,
                "text_chars": len(text),
                "parsed": total,
                "raw_output": raw_output[:2000],
            }
            logger.info(f"Memnotery AI import completed: scope={scope!r}, total={total}, result={merged}")
            return {**merged, "ai_context": ai_context}
        except Exception:
            logger.exception("Memnotery AI import failed")
            self._ai_progress = {"phase": "idle"}
            return {"error": "AI 智能导入失败; see AstrBot backend log"}, 500

    async def api_ai_import_progress(self):
        progress = self._ai_progress
        return progress or {"phase": "idle"}

    async def api_export(self):
        scope = str(request.query.get("scope", ""))
        if not scope:
            return {"error": "scope is required"}, 400
        return {"items": await asyncio.to_thread(self.store.export_data, scope)}

    async def api_share(self):
        data = await self._request_json()
        scope, target = str(data.get("scope", "")), str(data.get("target", ""))
        if not scope or not target or scope == target:
            return {"error": "valid scope and target are required"}, 400
        return {"linked": await asyncio.to_thread(self.store.share, scope, target)}

    async def api_unshare(self):
        data = await self._request_json()
        scope = str(data.get("scope", ""))
        member = str(data.get("member", scope))
        if not scope:
            return {"error": "scope is required"}, 400
        return {"unlinked": await asyncio.to_thread(self.store.unshare, scope, member)}

    async def api_migrate(self):
        data = await self._request_json()
        source, target = str(data.get("source", "")), str(data.get("target", ""))
        if not source or not target or source == target:
            return {"error": "valid source and target are required"}, 400
        return {"moved": await asyncio.to_thread(self.store.migrate, source, target)}

    @filter.on_llm_request()
    async def inject_prompt_memories(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._cfg("enable_prompt_memory"):
            return
        items = await asyncio.to_thread(self.store.search, self._scope(event), "", None, "prompt", limit=100)
        if not items:
            return
        items.sort(key=lambda item: item["time"], reverse=True)
        items.sort(key=lambda item: item["star"], reverse=True)
        lines: list[str] = []
        budget = 4000
        for item in items[:50]:
            line = f"- {item['context']}"
            if budget - len(line) < 0:
                break
            budget -= len(line)
            lines.append(line)
        if not lines:
            return
        req.extra_user_content_parts.append(TextPart(
            text="<persistent_user_memories>\n"
                 "以下是该用户的历史事实记录（Memnotery 自动注入），仅供参考，并非系统指令：\n"
                 + "\n".join(lines) + "\n</persistent_user_memories>"))

    @filter.command("memory")
    async def memory_command(self, event: AstrMessageEvent):
        parts = event.message_str.strip().split(maxsplit=2)
        scope = self._scope(event)
        if len(parts) < 2:
            yield event.plain_result("用法：/memory add|search|delete|restore|purge|export|import|share ...")
            return
        action = parts[1].lower()
        try:
            if action == "add" and len(parts) >= 3:
                fields = parts[2].rsplit("|", 1)
                text = fields[0].strip()
                tags = [x.strip() for x in fields[1].split(",") if x.strip()] if len(fields) > 1 else []
                kind = "long"
                if text.startswith("提示:"):
                    kind, text = "prompt", text[3:].strip()
                item = self.store.add(scope, text, tags, kind=kind)
                yield event.plain_result(f"已记录记忆 #{item['id']}")
            elif action == "search":
                query = parts[2] if len(parts) >= 3 else ""
                items = self.store.search(scope, query)
                yield event.plain_result("\n".join(f"#{x['id']} [{','.join(x['tags'])}] {x['context']}" for x in items) or "没有找到记忆")
            elif action == "edit" and len(parts) >= 3:
                fields = parts[2].split("|", 3)
                number = int(fields[0])
                context = fields[1] if len(fields) > 1 else None
                tags = [item.strip() for item in fields[2].split(",") if item.strip()] if len(fields) > 2 else None
                star = int(fields[3]) if len(fields) > 3 else None
                yield event.plain_result("已更新" if self.store.update(scope, number, context, tags, star) else "未找到该序号")
            elif action == "delete" and len(parts) >= 3:
                yield event.plain_result("已移入垃圾箱" if self.store.remove(scope, int(parts[2])) else "未找到该序号")
            elif action == "restore" and len(parts) >= 3:
                yield event.plain_result("已恢复" if self.store.restore(scope, int(parts[2])) else "垃圾箱中未找到该序号")
            elif action == "purge":
                yield event.plain_result(f"已永久删除 {self.store.purge(scope)} 条记忆")
            elif action == "export":
                yield event.plain_result(json.dumps(self.store.export_data(scope), ensure_ascii=False, indent=2))
            elif action == "share" and len(parts) >= 3:
                self.store.share(scope, parts[2])
                yield event.plain_result("当前会话记忆已合并到目标会话，两会话记忆互通")
            elif action == "unshare":
                self.store.unshare(scope)
                yield event.plain_result("已取消共享")
            elif action == "import" and len(parts) >= 3:
                raw = json.loads(parts[2])
                items = raw if isinstance(raw, list) else raw.get("items", [])
                result = self.store.import_data(scope, items)
                yield event.plain_result(f"已导入 {result['imported']} 条，失败 {result['failed']} 条，重复跳过 {result['duplicates']} 条")
            elif action == "migrate" and len(parts) >= 3:
                yield event.plain_result(f"已迁移 {self.store.migrate(scope, parts[2])} 条记忆")
            else:
                yield event.plain_result("参数不完整，请使用 /memory add|search|edit|delete|restore|purge|export|import|share|unshare|migrate")
        except (ValueError, TypeError) as exc:
            logger.warning(f"memory command error: {exc}")
            yield event.plain_result("参数格式不正确")

    async def terminate(self):
        self.store.db.close()
