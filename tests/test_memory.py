"""Tests for MemoryStore and InMemoryStore."""

import pytest
from agentx.memory.in_memory import InMemoryStore, MemoryStoreUnavailableError


def test_in_memory_store_save_and_load():
    store = InMemoryStore()
    record = store.save_context("agent_1", {"key": "value"}, conversation_id="conv_1")

    assert record.agent_id == "agent_1"
    assert record.conversation_id == "conv_1"
    assert record.data == {"key": "value"}

    loaded = store.load_context("agent_1", conversation_id="conv_1")
    assert len(loaded) == 1
    assert loaded[0].data["key"] == "value"


def test_in_memory_store_retry_and_failure():
    store = InMemoryStore(max_retries=2, retry_delay=0.01)
    store.is_available = False

    with pytest.raises(MemoryStoreUnavailableError):
        store.save_context("agent_1", {"test": 123})


def test_in_memory_store_recovery():
    store = InMemoryStore(max_retries=3, retry_delay=0.01)
    store.is_available = True
    record = store.save_context("agent_1", {"recovered": True})
    assert record is not None
