from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal

import psycopg
from pgvector.psycopg import register_vector

from .config import settings


@contextmanager
def get_conn():
    """每次连接都注册 vector 类型，否则读写 embedding 会报类型错误。"""
    conn = psycopg.connect(settings.pg_dsn)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _plain(value):
    # NUMERIC 列（price）在 psycopg 里是 Decimal：json.dumps 会直接报
    # "Object of type Decimal is not JSON serializable"，内存后端却是 float。
    # 在 DB 边界统一成 float，两条召回路的数据形状才一致。
    if isinstance(value, Decimal):
        return float(value)
    return value


def rows_to_dicts(cur) -> list[dict]:
    cols = [d.name for d in cur.description]
    return [{k: _plain(v) for k, v in zip(cols, row)} for row in cur.fetchall()]
