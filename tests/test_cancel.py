"""「停止生成」的取消链路：取消令牌要能中断 LLM 流并断开模型连接。"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import shopagent.llm as llm
from shopagent.agent import graph


def _chunk(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


class _FakeStream:
    """第一段吐完后设置取消事件，模拟用户看到首字后点停止。"""

    def __init__(self, chunks, cancel):
        self.chunks = list(chunks)
        self.cancel = cancel
        self.closed = False

    def __iter__(self):
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if index == 0:
                self.cancel.set()

    def close(self):
        self.closed = True


def test_chat_stream_cancel_stops_generation(monkeypatch):
    cancel = threading.Event()
    stream = _FakeStream([_chunk("你"), _chunk("好")], cancel)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: stream)
        )
    )
    monkeypatch.setattr(llm, "_client", lambda: fake_client)
    monkeypatch.setattr(llm, "_model_id", lambda: "fake-model")

    with llm.cancel_token(cancel):
        gen = llm.chat_stream([{"role": "user", "content": "hi"}])
        assert next(gen) == "你"
        with pytest.raises(llm.GenerationCancelled):
            next(gen)

    # 取消时主动断开与模型的连接，避免后台继续生成
    assert stream.closed is True


class _EndlessGraph:
    """模拟一轮还在跑的 Agent：每 50ms 产出一个 trace，直到被取消。"""

    def stream(self, state, stream_mode):
        index = 0
        while True:
            index += 1
            yield {"trace": [f"step {index}"]}
            time.sleep(0.05)


def test_stream_agent_releases_lock_when_consumer_disconnects(monkeypatch):
    monkeypatch.setattr(graph, "GRAPH", _EndlessGraph())
    lock = threading.Lock()

    agent_iter = graph.stream_agent("hi", profile={}, lock=lock)
    assert next(agent_iter)[0] == "step"

    # 客户端断开：生成器被 close，worker 必须自己把锁还回来
    agent_iter.close()

    assert lock.acquire(blocking=False) is True
    lock.release()


def test_cancelled_token_blocks_new_llm_calls(monkeypatch):
    cancel = threading.Event()
    cancel.set()

    def boom():
        raise AssertionError("取消后不应该再发起 LLM 调用")

    monkeypatch.setattr(llm, "_client", boom)

    with llm.cancel_token(cancel):
        with pytest.raises(llm.GenerationCancelled):
            llm.chat([{"role": "user", "content": "hi"}])
