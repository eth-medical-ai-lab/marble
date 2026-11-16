"""BLOX agent observations, action parsing, and feedback."""
from __future__ import annotations
import re
from typing import List, Optional, Any, Dict
from blox.constants import Action, ACTION_FROM_NAME, ACTION_NAMES
from agent_loop import AgentEnv, AgentLoopState, StepOutcome
from data_paths import public_data_path
from blox.env import BloxEnv, has_crumble, has_fragile, validate_solution
from blox.rendering import render_b64
from eval_utils import image_content_block
from blox.config import BloxEnvConfig
from blox.prompts import (
    board_to_text,
    build_blox_continue_feedback_text,
    build_blox_fall_feedback_text,
    build_blox_parse_error_text,
    build_blox_success_feedback_text,
    build_instruction_prompt,
    build_observation_prompt,
    state_description,
)


def parse_action_sequence(text: str) -> List[Action]:
    """
    Parse a sequence of actions (UP/DOWN/LEFT/RIGHT) from LLM text output.

    Strategy: strip thinking blocks, try to isolate a dedicated answer section
    first (to avoid picking up direction words from chain-of-thought reasoning),
    then fall back to scanning the full text.

    Recognised answer sections (case-insensitive):
        **Solution:** LEFT, RIGHT, UP
        ```\nLEFT, RIGHT\n```
        Answer: UP, DOWN
        Final answer: RIGHT
        Plan: UP → DOWN → LEFT
    """
    from eval_utils import strip_thinking_blocks

    text = strip_thinking_blocks(text)
    action_pattern = r"\b(UP|DOWN|LEFT|RIGHT)\b"

    answer_block = _extract_answer_block(text)
    if answer_block is not None:
        matches = re.findall(action_pattern, answer_block.upper())
        if matches:
            return [ACTION_FROM_NAME[m] for m in matches if m in ACTION_FROM_NAME]

    matches = re.findall(action_pattern, text.upper())
    return [ACTION_FROM_NAME[m] for m in matches if m in ACTION_FROM_NAME]


def _extract_answer_block(text: str) -> Optional[str]:
    """Try to isolate the final answer section from LLM output."""
    action_re = r"\b(UP|DOWN|LEFT|RIGHT)\b"

    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^}]*\})+)\}", text)
    if boxed:
        last_boxed = re.sub(r"\\text\{([^}]*)\}", r"\1", boxed[-1]).strip()
        if re.search(action_re, last_boxed, re.IGNORECASE):
            return last_boxed

    code_blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        block = code_blocks[-1].strip()
        if re.search(action_re, block, re.IGNORECASE):
            return block

    label_pat = re.compile(
        r"(?:^|\n)\s*\**\s*(?:the\s+)?(?:final\s+|next\s+(?:single\s+)?)?"
        r"(?:solution|answer|plan|moves?)\s*(?:is)?\s*:?\s*\**\s*",
        re.IGNORECASE,
    )
    label_matches = list(label_pat.finditer(text))
    if label_matches:
        last = label_matches[-1]
        block = text[last.end():].strip()
        if re.search(action_re, block, re.IGNORECASE):
            return block

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if re.fullmatch(r"\**\s*" + action_re + r"\s*\**", last_line, re.IGNORECASE):
            return last_line

    return None


class BloxAgentEnv(AgentEnv):
    """AgentEnv wrapper for the Blox multi-step task."""

    def __init__(
        self,
        env_cfg: BloxEnvConfig,
        level_path: Optional[str] = None,
        level_data: Optional[Dict[str, Any]] = None,
        level_id: Optional[str] = None,
        loop_mode: str = "multi_turn_conversation",
    ) -> None:
        super().__init__(loop_mode=loop_mode)
        self.env_cfg = env_cfg
        self.level_path = level_path
        self.level_data = level_data
        self.level_id = level_id or (public_data_path(level_path) if level_path else "level")
        self.env: Optional[BloxEnv] = None
        self.actions_taken: List[str] = []
        self.success: bool = False
        self.failure_reason: Optional[str] = None
        self._start_pos: Optional[tuple[int, int]] = None
        self.offline_actions: List[str] = []
        self.offline_steps_completed: int = 0
        self.offline_message: str = ""

    @property
    def max_steps(self) -> int:
        return int(self.env_cfg.max_steps)

    def _build_observation_content(self) -> List[Dict[str, Any]]:
        assert self.env is not None
        state = self.env._state
        content: List[Dict[str, Any]] = []
        if self.env_cfg.with_image:
            b64 = render_b64(state, start_pos=self._start_pos)
            content.append(image_content_block(b64))
        if self.env_cfg.with_symbolic:
            text = board_to_text(
                state.board,
                state.block,
                state.goal_r,
                state.goal_c,
                active_bridges=state.active_bridges,
                current_height=state.current_height,
                collapsed_crumble=state.collapsed_crumble,
            )
            if not self.env_cfg.minimal_obs:
                text += "\n" + state_description(state)
            content.append({"type": "text", "text": text})
        return content

    def _build_initial_prompt_text(self, mode: str) -> str:
        """Build the first-turn user prompt shared across all Blox modes."""
        assert self.env is not None
        state = self.env._state
        instruction_prompt = build_instruction_prompt(
            has_switches=bool(state.board.switches),
            has_mixed_bridges=bool(state.active_bridges),
            has_fragile=has_fragile(state.board),
            has_crumble=has_crumble(state.board),
            has_hard_switches=bool(state.board.hard_switches),
            has_elevators=bool(state.board.elevators),
            move_budget=self.env_cfg.move_budget,
            max_steps=self.max_steps,
            mode=mode,
        )
        observation_prompt = build_observation_prompt(
            state,
            include_image=self.env_cfg.with_image,
            include_symbolic=self.env_cfg.with_symbolic,
            minimal_obs=self.env_cfg.minimal_obs,
        )

        combined_text = instruction_prompt
        if observation_prompt:
            combined_text += "\n\n" + observation_prompt
        return combined_text

    def _build_initial_messages(self, mode: str) -> List[Dict[str, Any]]:
        assert self.env is not None
        state = self.env._state
        content: List[Dict[str, Any]] = []
        if self.env_cfg.with_image:
            b64 = render_b64(state, start_pos=self._start_pos)
            content.append(image_content_block(b64))

        combined_text = self._build_initial_prompt_text(mode)
        self.last_prompt_text = combined_text
        content.append({"type": "text", "text": combined_text})
        return [{"role": "user", "content": content}]

    def reset(self, run_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        self.env = BloxEnv(level=self.level_data, level_path=self.level_path, max_steps=self.max_steps)
        self.env.reset()
        self._start_pos = (self.env._state.block.r, self.env._state.block.c)
        self.actions_taken = []
        self.success = False
        self.failure_reason = None
        self.offline_actions = []
        self.offline_steps_completed = 0
        self.offline_message = ""

        return self._build_initial_messages(self.loop_mode())

    def step_multi_turn_conversation(
        self,
        response_text: str,
        step_idx: int,
        run_dir: Optional[str] = None,
    ) -> StepOutcome:
        assert self.env is not None

        parsed = parse_action_sequence(response_text)
        if not parsed:
            feedback = build_blox_parse_error_text(self.loop_mode())
            return self._build_text_step_outcome(
                feedback,
                metadata={"status": "parse_error"},
            )

        action = parsed[0]
        self.actions_taken.append(ACTION_NAMES[action])
        _, _, terminated, truncated, step_info = self.env.step(int(action))

        observation_parts: List[Dict[str, Any]] = []
        collapsed_cells = step_info.get("collapsed_crumble", [])
        collapsed_text = None
        if collapsed_cells:
            coords = ", ".join(f"(x={c}, y={r})" for r, c in collapsed_cells)
            collapsed_text = f"CRUMBLE tiles collapsed: {coords}"
        if step_info.get("fell"):
            observation_parts = self._build_observation_content()
            remaining = self.max_steps - step_idx - 1
            feedback_text = build_blox_fall_feedback_text(
                action_name=ACTION_NAMES[action],
                actions_taken=self.actions_taken,
                remaining_moves=remaining,
                loop_mode=self.loop_mode(),
                env_feedback=step_info.get("feedback"),
                collapsed_text=collapsed_text,
            )
            feedback_parts = self._build_followup_content(
                observation=observation_parts,
                feedback_text=feedback_text,
            )
        elif step_info.get("won"):
            feedback_text = build_blox_success_feedback_text(
                action_name=ACTION_NAMES[action],
                step_idx=step_idx,
                collapsed_text=collapsed_text,
            )
            feedback_parts = self._build_followup_content(feedback_text=feedback_text)
            self.success = True
        elif step_info.get("dead_end"):
            self.failure_reason = step_info.get("failure_reason", "unreachable_after_crumble")
            lines = [
                step_info.get("feedback")
                or "The puzzle can no longer be completed from the current state."
            ]
            if collapsed_text:
                lines.append(collapsed_text)
            feedback_parts = self._build_followup_content(
                feedback_text="\n".join(lines),
            )
        else:
            observation_parts = self._build_observation_content()
            remaining = self.max_steps - step_idx - 1
            feedback_text = build_blox_continue_feedback_text(
                action_name=ACTION_NAMES[action],
                step_idx=step_idx,
                actions_taken=self.actions_taken,
                remaining_moves=remaining,
                loop_mode=self.loop_mode(),
                collapsed_text=collapsed_text,
            )
            feedback_parts = self._build_followup_content(
                observation=observation_parts,
                feedback_text=feedback_text,
            )

        return self._build_step_outcome(
            content=feedback_parts,
            action_summary=ACTION_NAMES[action],
            metadata={
                "status": "dead_end" if step_info.get("dead_end")
                else "continue" if not (terminated or truncated)
                else "done",
                **(
                    {"failure_reason": self.failure_reason}
                    if self.failure_reason
                    else {}
                ),
            },
            done=bool(terminated or truncated),
            success=self.success,
        )

    def step_single_turn(
        self,
        response_text: str,
        reasoning_text: str = "",
        run_dir: Optional[str] = None,
    ) -> None:
        """Validate a full offline move sequence against the initial level state."""
        del run_dir
        self.last_response_text = response_text or ""
        self.last_reasoning_text = reasoning_text or ""

        actions = parse_action_sequence(self.last_response_text)
        if not actions and reasoning_text:
            actions = parse_action_sequence(reasoning_text)

        if self.env_cfg.move_budget is not None:
            actions = actions[: int(self.env_cfg.move_budget)]

        success, steps_completed, message = validate_solution(self.env._state, actions)
        self.success = bool(success)
        self.offline_steps_completed = int(steps_completed)
        self.offline_message = str(message)
        self.offline_actions = [ACTION_NAMES[action] for action in actions]

    def build_result(
        self,
        loop_state: AgentLoopState,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.loop_mode() == "single_turn_offline":
            usage = None
            if loop_state.usage_log:
                usage = loop_state.usage_log[0].get("usage")
            return {
                "level_id": self.level_id,
                "success": self.success,
                "steps_completed": self.offline_steps_completed,
                "message": self.offline_message,
                "move_budget": self.env_cfg.move_budget,
                "predicted_actions": list(self.offline_actions),
                "num_predicted": len(self.offline_actions),
                "usage": usage,
                "response": self.last_response_text,
                "finish_reason": self.last_finish_reason,
            }

        result = {
            "level_id": self.level_id,
            "success": self.success,
            "actions_taken": list(self.actions_taken),
            "num_steps": len(self.actions_taken),
            "usage": loop_state.usage_log,
        }
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason
        return result
