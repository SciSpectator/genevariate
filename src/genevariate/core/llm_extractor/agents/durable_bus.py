"""Durable variant of :pyclass:`agents.base.MessageBus`.

Each :pyclass:`AgentMessage` is persisted to
:pyclass:`gse_context_cache.GSEContextCache.agent_messages` *before* it
is enqueued into the in-process inbox; the row is acked after the
recipient processes it (via ``DurableMessageBus.ack``). On bus startup,
``recover()`` re-enqueues every undelivered or delivered-but-unacked
message so a crashed agent's work resumes automatically.

In-process delivery semantics are unchanged: callers use ``post`` /
``call`` / ``reply`` exactly like the in-memory bus. The persistence is
additive, low-overhead (single insert per post + one update per ack).

Wire into the existing fleet by passing a ``DurableMessageBus`` instance
to :pyclass:`agents.coordinator.Coordinator(bus=...)` instead of the
default ``MessageBus()``.
"""
from __future__ import annotations

import time
from typing import Optional

from .base import AgentMessage, MessageBus

# Imported lazily so the agents package doesn't pull sqlite at import
# time when the durable bus isn't used.
def _default_cache():
    from gse_context_cache import GSEContextCache
    return GSEContextCache()


class DurableMessageBus(MessageBus):
    """``MessageBus`` with sqlite-backed inbox/outbox for crash safety."""

    def __init__(self, cache=None):
        super().__init__()
        self.cache = cache or _default_cache()
        # Map correlation_id -> persisted row id, so ``ack`` can find it
        # without re-deriving from the message.
        self._row_for_corr: dict[str, int] = {}
        # Idempotency guard: protects against accidental double-recover
        # during re-init. Set to True after the first ``recover()`` call.
        self._recovered: bool = False
        # Last persist failure (None on success). Useful for callers
        # that want to know whether a post is durable or in-memory only.
        self.last_persist_error: Optional[BaseException] = None

    # ── persistence shims ────────────────────────────────────────────────
    def _persist(self, msg: AgentMessage) -> int:
        payload = {
            "method": msg.method,
            "params": msg.params or {},
            "reply_to": msg.reply_to,
            "ts": msg.ts,
            "result": msg.result,
            "error":  msg.error,
        }
        row_id = self.cache.post_message(
            sender=msg.sender or None,
            recipient=msg.recipient,
            kind=msg.method,
            payload=payload,
            correlation=msg.correlation_id,
        )
        self._row_for_corr[msg.correlation_id] = row_id
        return row_id

    # ── overridden public API ────────────────────────────────────────────
    def post(self, msg: AgentMessage) -> None:
        # Persist first so a crash between persist and enqueue is safe;
        # recover() will re-enqueue.
        try:
            self._persist(msg)
            self.last_persist_error = None
        except Exception as e:                                  # pragma: no cover
            self.last_persist_error = e
            print(f"[durable_bus] persist failed (continuing): {e!r}", flush=True)
        super().post(msg)

    def ack(self, msg: AgentMessage) -> None:
        """Mark ``msg`` as fully processed in the durable log."""
        row = self._row_for_corr.pop(msg.correlation_id, None)
        if row is None:
            return
        try:
            self.cache.ack_message(row)
        except Exception as e:                                  # pragma: no cover
            print(f"[durable_bus] ack failed: {e!r}", flush=True)

    # ── crash recovery ───────────────────────────────────────────────────
    def recover(self, recipients: Optional[list[str]] = None,
                redeliver: bool = True,
                force: bool = False) -> int:
        """Re-enqueue undelivered (or unacked-and-delivered) messages.

        ``recipients`` defaults to every registered inbox. ``redeliver``
        also pulls already-delivered-but-not-acked rows back into the
        inbox (typical agent-crash scenario).

        Idempotent: a second call returns 0 unless ``force=True`` is
        passed. Pass ``force=True`` only if you have torn down all
        agents and re-constructed the bus from scratch.

        Returns the number of envelopes recovered.
        """
        if self._recovered and not force:
            return 0
        if recipients is None:
            recipients = list(self._inboxes.keys())
        n = 0
        for rec in recipients:
            # First: undelivered (delivered_at IS NULL). fetch_inbox marks
            # them delivered atomically, so ack() will clear them later.
            for env in self.cache.fetch_inbox(rec, limit=10_000):
                msg = AgentMessage(
                    method=env["kind"],
                    params=env["payload"].get("params", {}),
                    sender=env.get("sender") or "",
                    recipient=rec,
                    correlation_id=env["correlation"]
                                   or f"row-{env['id']}",
                    reply_to=env["payload"].get("reply_to"),
                    ts=env["payload"].get("ts", env["created_at"]),
                )
                self._row_for_corr[msg.correlation_id] = env["id"]
                # Skip persist (already persisted) — go straight to inbox.
                MessageBus.post(self, msg)
                n += 1
            # Optional: redelivered-but-stuck (cluster restart).
            if redeliver:
                for r in self.cache.fetch_redelivery(rec):
                    payload = r["payload"]
                    msg = AgentMessage(
                        method=r["kind"], params=payload.get("params", {}),
                        sender=r["sender"] or "", recipient=rec,
                        correlation_id=r["correlation"]
                                       or f"row-{r['id']}",
                        reply_to=payload.get("reply_to"),
                        ts=payload.get("ts", r["created_at"]))
                    self._row_for_corr[msg.correlation_id] = r["id"]
                    MessageBus.post(self, msg)
                    n += 1
        self._recovered = True
        return n
