from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eval_utils import ModelConfig, aggregate_usage, write_conversation_json, write_conversation_txt
from llm_backends import ChatResult, LLMBackend, make_backend


Message = Dict[str, Any]


@dataclass
class StepOutcome:
    """Result of interpreting one assistant turn inside an AgentEnv."""

    user_messages: List[Message] = field(default_factory=list)
    action_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    success: bool = False


@dataclass
class AgentTurnRecord:
    """Structured record for a single LLM call inside AgentLoop."""

    step: int
    response_text: str
    usage: Optional[Dict[str, Any]]
    action_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentLoopState:
    """Shared runtime state captured by AgentLoop during an episode."""

    history: List[Message] = field(default_factory=list)
    turns: List[AgentTurnRecord] = field(default_factory=list)
    llm_usage: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    done: bool = False

    @property
    def usage_log(self) -> List[Dict[str, Any]]:
        return list(self.llm_usage)


class AgentEnv(ABC):
    """Agent-facing environment contract for the shared AgentLoop runtime."""

    def __init__(self, loop_mode: str = "multi_turn_conversation") -> None:
        self._loop_mode = loop_mode
        self.last_prompt_text: str = ""
        self.last_response_text: str = ""
        self.last_reasoning_text: str = ""
        self.last_finish_reason: Optional[str] = None
        self.last_usage: Optional[Dict[str, Any]] = None

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum number of LLM turns for the current episode."""

    def loop_mode(self) -> str:
        """Return either ``multi_turn_conversation`` or ``single_turn_offline``."""

        return self._loop_mode

    def _build_step_outcome(
        self,
        content: List[Message],
        action_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        done: bool = False,
        success: bool = False,
    ) -> StepOutcome:
        """Wrap follow-up feedback as the next user message."""

        return StepOutcome(
            user_messages=[{"role": "user", "content": content}],
            action_summary=action_summary,
            metadata=dict(metadata or {}),
            done=done,
            success=success,
        )

    @staticmethod
    def _text_content_block(text: str) -> Message:
        return {"type": "text", "text": text}

    @staticmethod
    def _join_text_blocks(content: List[Message]) -> str:
        text_blocks = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip()
        ]
        return "\n\n".join(text_blocks)

    def _build_followup_content(
        self,
        *,
        observation: Optional[List[Message]] = None,
        feedback_text: Optional[str] = None,
        terminal_text: Optional[str] = None,
    ) -> List[Message]:
        """Assemble the next-turn payload in a consistent order."""

        content = list(observation or [])
        if feedback_text:
            content.append(self._text_content_block(feedback_text))
        if terminal_text:
            content.append(self._text_content_block(terminal_text))
        self.last_prompt_text = self._join_text_blocks(content)
        return content

    def _build_text_step_outcome(
        self,
        text: str,
        action_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        done: bool = False,
        success: bool = False,
    ) -> StepOutcome:
        """Convenience wrapper for text-only follow-up feedback."""

        return self._build_step_outcome(
            content=self._build_followup_content(feedback_text=text),
            action_summary=action_summary,
            metadata=metadata,
            done=done,
            success=success,
        )

    @abstractmethod
    def reset(self, run_dir: Optional[str] = None) -> List[Message]:
        """Reset internal episode state and return initial history messages."""


    @abstractmethod
    def step_multi_turn_conversation(
        self,
        response_text: str,
        step_idx: int,
        run_dir: Optional[str] = None,
    ) -> StepOutcome:
        """Interpret one plain-text multi-turn assistant response."""

    @abstractmethod
    def build_result(
        self,
        loop_state: AgentLoopState,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the final result payload for the episode."""

    def step_single_turn(
        self,
        response_text: str,
        reasoning_text: str = "",
        run_dir: Optional[str] = None,
    ) -> None:
        """Interpret one single-turn offline response and update env state."""

        raise NotImplementedError(f"{type(self).__name__} does not implement step_single_turn()")

    def result_path(self, run_dir: str) -> str:
        return os.path.join(run_dir, "result.json")

    def load_cached_result(self, run_dir: Optional[str]) -> Optional[Dict[str, Any]]:
        if not run_dir:
            return None
        path = self.result_path(run_dir)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_result(self, result: Dict[str, Any], run_dir: Optional[str]) -> None:
        if not run_dir:
            return
        os.makedirs(run_dir, exist_ok=True)
        with open(self.result_path(run_dir), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


class AgentLoop:
    """Shared runtime for single-shot and plain multi-turn evaluations."""

    def __init__(self, model_cfg: ModelConfig, backend: Optional[LLMBackend] = None) -> None:
        self.model_cfg = model_cfg
        self.backend = backend

    def _write_usage(self, run_dir: Optional[str], loop_state: AgentLoopState) -> None:
        if not run_dir:
            return
        os.makedirs(run_dir, exist_ok=True)
        usage_data = {
            "steps": loop_state.usage_log,
            "total": aggregate_usage([item["usage"] for item in loop_state.usage_log]),
        }
        with open(os.path.join(run_dir, "usage.json"), "w", encoding="utf-8") as f:
            json.dump(usage_data, f, indent=2)

    def _backend(self) -> LLMBackend:
        if self.backend is None:
            self.backend = make_backend(
                self.model_cfg.model,
                local_server_port=self.model_cfg.local_server_port,
                local_server_url=self.model_cfg.local_server_url,
            )
        return self.backend

    @staticmethod
    def _build_assistant_entry(result: ChatResult) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "role": "assistant",
            "content": result.content or "",
        }
        if result.finish_reason:
            entry["finish_reason"] = result.finish_reason
        if result.reasoning_content:
            entry["reasoning_content"] = result.reasoning_content
        return entry

    def _chat(self, messages: List[Message]) -> ChatResult:
        """Request one response per turn, without hidden continuations or retries."""
        return self._backend().chat(
            messages=messages,
            model=self.model_cfg.model,
            max_tokens=self.model_cfg.max_tokens,
            reasoning_effort=self.model_cfg.reasoning_effort,
        )

    @staticmethod
    def _append_outcome_messages(state: AgentLoopState, outcome: StepOutcome) -> None:
        for msg in outcome.user_messages:
            user_msg = dict(msg)
            user_msg.setdefault("role", "user")
            state.history.append(user_msg)

    def _run_single_turn(
        self,
        agent_env: AgentEnv,
        state: AgentLoopState,
        run_dir: Optional[str] = None,
    ) -> None:
        result = self._chat(messages=list(state.history))
        response_text = result.content
        reasoning_text = result.reasoning_content or ""
        usage = result.usage
        if usage is not None:
            state.llm_usage.append({"step": 0, "usage": usage})

        state.history.append(self._build_assistant_entry(result))

        agent_env.last_response_text = response_text
        agent_env.last_reasoning_text = reasoning_text
        agent_env.last_finish_reason = result.finish_reason
        agent_env.last_usage = usage

        state.turns.append(AgentTurnRecord(step=0, response_text=response_text, usage=usage))
        agent_env.step_single_turn(
            response_text=response_text,
            reasoning_text=reasoning_text,
            run_dir=run_dir,
        )
        state.success = bool(agent_env.build_result(state, run_dir=run_dir).get("success", False))
        state.done = True

    def _run_multi_turn_conversation(
        self,
        agent_env: AgentEnv,
        state: AgentLoopState,
        run_dir: Optional[str] = None,
    ) -> None:
        for step_idx in range(agent_env.max_steps):
            result = self._chat(messages=list(state.history))
            usage = result.usage
            reasoning_text = result.reasoning_content or ""
            response_text = result.content
            if usage is not None:
                state.llm_usage.append({"step": step_idx, "usage": usage})

            state.history.append(self._build_assistant_entry(result))

            agent_env.last_response_text = response_text
            agent_env.last_reasoning_text = reasoning_text
            agent_env.last_finish_reason = result.finish_reason
            agent_env.last_usage = usage
            turn = AgentTurnRecord(step=step_idx, response_text=response_text, usage=usage)
            state.turns.append(turn)

            outcome = agent_env.step_multi_turn_conversation(
                response_text=response_text,
                step_idx=step_idx,
                run_dir=run_dir,
            )
            turn.action_summary = outcome.action_summary
            turn.metadata = dict(outcome.metadata)
            self._append_outcome_messages(state, outcome)

            if outcome.success:
                state.success = True
            if outcome.done:
                state.done = True
                break

    def run(self, agent_env: AgentEnv, run_dir: Optional[str] = None) -> Dict[str, Any]:
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            cached = agent_env.load_cached_result(run_dir)
            if cached is not None:
                return cached

        init_messages = list(agent_env.reset(run_dir))
        state = AgentLoopState(history=init_messages)
        loop_mode = agent_env.loop_mode()

        if loop_mode == "single_turn_offline":
            self._run_single_turn(agent_env, state, run_dir=run_dir)
        elif loop_mode == "multi_turn_conversation":
            self._run_multi_turn_conversation(agent_env, state, run_dir=run_dir)
        else:
            raise ValueError(f"Unknown agent loop mode: {loop_mode}")

        self._write_usage(run_dir, state)
        if run_dir:
            write_conversation_json(state.history, run_dir)
            write_conversation_txt(state.history, run_dir)
        result = agent_env.build_result(state, run_dir=run_dir)
        agent_env.save_result(result, run_dir)
        return result
