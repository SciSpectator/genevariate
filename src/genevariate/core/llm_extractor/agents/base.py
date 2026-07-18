"""A2A primitives: message envelope, message bus, agent base class.

The bus is intentionally thin so the same agent code runs against:

    - in-process queues  (this default — no extra deps)
    - Redis Streams      (cluster fan-out; subclass MessageBus)
    - native A2A         (cross-org interop; subclass MessageBus)

Agents are message-driven workers. Each agent advertises a capability
(``Tissue.normalize`` etc.) via ``capabilities()``; the coordinator
routes ``AgentMessage(method=...)`` to whichever agent claims that
method. Reply-to fan-in is request/response over the same bus.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AgentMessage:
    """Envelope passed between agents.

    Fields mirror what an A2A / JSON-RPC payload would carry, so the
    in-process bus can be swapped for a real one without changing
    agent logic.
    """
    method:         str
    params:         Dict[str, Any] = field(default_factory=dict)
    sender:         str            = ""
    recipient:      str            = ""
    correlation_id: str            = field(default_factory=lambda: str(uuid.uuid4()))
    reply_to:       Optional[str]  = None
    ts:             float          = field(default_factory=time.time)
    # Reply payload — populated by the responder.
    result:         Optional[Dict[str, Any]] = None
    error:          Optional[str]            = None


class MessageBus:
    """In-process queue-backed message bus.

    Each registered agent gets its own inbox. Senders ``post(msg)``;
    the bus routes by ``msg.recipient``. For request/reply the sender
    waits on a ``threading.Event`` keyed by ``correlation_id`` and the
    bus signals it on response delivery.
    """

    def __init__(self) -> None:
        self._inboxes: Dict[str, "queue.Queue[AgentMessage]"] = {}
        self._pending: Dict[str, threading.Event] = {}
        self._results: Dict[str, AgentMessage] = {}
        self._lock = threading.Lock()
        # method -> agent name routing table built from capabilities().
        self._routes: Dict[str, str] = {}

    # ── registration ────────────────────────────────────────────────────
    def register_inbox(self, name: str) -> None:
        with self._lock:
            self._inboxes.setdefault(name, queue.Queue())

    def register_routes(self, agent_name: str, methods: List[str]) -> None:
        with self._lock:
            for m in methods:
                self._routes[m] = agent_name

    def register(self, agent: "AgentBase") -> None:
        """Back-compat shim — registers inbox AND routes in one call.
        Subclasses populating ``handlers`` after super().__init__ should
        instead call ``register_inbox`` in __init__ and
        ``register_routes`` later (the agent base does this via
        ``_finalize_routes``).
        """
        self.register_inbox(agent.name)
        self.register_routes(agent.name, list(agent.capabilities()))

    def inbox(self, name: str) -> "queue.Queue[AgentMessage]":
        return self._inboxes[name]

    # ── send / receive ──────────────────────────────────────────────────
    def post(self, msg: AgentMessage) -> None:
        """Drop ``msg`` into the recipient's inbox. If recipient is empty,
        route by method via the capability table.
        """
        if not msg.recipient:
            msg.recipient = self._routes.get(msg.method, "")
        if not msg.recipient or msg.recipient not in self._inboxes:
            raise KeyError(
                f"MessageBus.post: no route for method={msg.method!r} "
                f"recipient={msg.recipient!r}"
            )
        self._inboxes[msg.recipient].put(msg)

    def call(self, msg: AgentMessage, timeout: float = 60.0) -> AgentMessage:
        """Synchronous request/reply over the bus. Sender blocks on a
        per-correlation-id event; recipient calls ``reply()`` which
        wakes the sender and returns the response envelope.
        """
        ev = threading.Event()
        with self._lock:
            self._pending[msg.correlation_id] = ev
        msg.reply_to = msg.sender or "__sync_caller__"
        self.post(msg)
        if not ev.wait(timeout):
            with self._lock:
                self._pending.pop(msg.correlation_id, None)
            raise TimeoutError(
                f"MessageBus.call: no reply for {msg.method!r} within {timeout}s"
            )
        with self._lock:
            return self._results.pop(msg.correlation_id)

    def reply(self, original: AgentMessage,
              result: Optional[Dict[str, Any]] = None,
              error:  Optional[str]            = None) -> None:
        """Send a response to a synchronous caller (or post to reply_to)."""
        resp = AgentMessage(
            method=original.method + ".reply",
            sender=original.recipient,
            recipient=original.reply_to or "",
            correlation_id=original.correlation_id,
            result=result, error=error,
        )
        with self._lock:
            ev = self._pending.pop(original.correlation_id, None)
            if ev is not None:
                self._results[original.correlation_id] = resp
                ev.set()
                return
        # Async reply path (no waiter) — drop into recipient's inbox.
        if resp.recipient and resp.recipient in self._inboxes:
            self._inboxes[resp.recipient].put(resp)


# ─────────────────────────────────────────────────────────────────────────────
# Agent base
# ─────────────────────────────────────────────────────────────────────────────
class AgentBase(threading.Thread):
    """Worker thread consuming its inbox and dispatching to handlers.

    Subclasses register handlers via ``self.handlers`` mapping
    ``method -> callable(msg) -> dict | None``. Returning a dict
    triggers an automatic reply; returning None means fire-and-forget.
    """

    def __init__(self, name: str, bus: MessageBus) -> None:
        super().__init__(name=name, daemon=True)
        self.name = name
        self.bus  = bus
        if not hasattr(self, "handlers"):
            self.handlers: Dict[str, Callable[[AgentMessage], Optional[Dict[str, Any]]]] = {}
        self._stop_evt = threading.Event()
        # Inbox first (so agents can be addressed by name even if
        # they haven't published handlers yet); routes are finalized
        # after the subclass populates ``handlers`` — see _finalize_routes.
        bus.register_inbox(name)
        self._finalize_routes()

    def _finalize_routes(self) -> None:
        """Publish current handler keys to the bus's routing table.
        Subclasses that populate ``self.handlers`` AFTER super().__init__
        should call this again at the end of their own __init__."""
        self.bus.register_routes(self.name, list(self.handlers))

    # ── subclass hooks ──────────────────────────────────────────────────
    def capabilities(self) -> List[str]:
        """Return the methods this agent serves. Override in subclass."""
        return list(self.handlers)

    # ── lifecycle ───────────────────────────────────────────────────────
    def stop(self) -> None:
        self._stop_evt.set()
        # Wake the inbox by posting a sentinel.
        self.bus.inbox(self.name).put(
            AgentMessage(method="__stop__", recipient=self.name)
        )

    def run(self) -> None:
        # Lazy import: tracing is optional. If the module / Phoenix isn't
        # available the agent still runs (no spans emitted).
        try:
            from .tracing import agent_span
        except Exception:                              # noqa: BLE001
            agent_span = None                          # type: ignore[assignment]

        inbox = self.bus.inbox(self.name)
        while not self._stop_evt.is_set():
            try:
                msg = inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg.method == "__stop__":
                break
            handler = self.handlers.get(msg.method)
            if handler is None:
                self.bus.reply(msg, error=f"no handler for {msg.method!r}")
                continue
            try:
                if agent_span is not None:
                    with agent_span(self.name, msg.method, msg.params) as ctx:
                        result = handler(msg)
                        ctx["output"] = result
                else:
                    result = handler(msg)
            except Exception as e:                     # noqa: BLE001
                self.bus.reply(msg, error=f"{type(e).__name__}: {e}")
                continue
            if result is not None or msg.reply_to:
                self.bus.reply(msg, result=result or {})
