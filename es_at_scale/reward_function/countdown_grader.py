import re
from typing import Any, Dict, List, Optional, Tuple, Union


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """Score for the <think>...</think><answer>...</answer> format.

    1.0 if the full format matches exactly, otherwise a partial credit
    of 0.1 (think present) + 0.5 (answer present).
    """
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    if re.match(full_format_regex, response, re.DOTALL):
        return 1.0

    reward = 0.0
    if re.search(think_regex, response, re.DOTALL):
        reward += 0.1
    if re.search(answer_regex, response, re.DOTALL):
        reward += 0.5
    return reward


def answer_reward_function(response: str, numbers: List[int], target: int) -> float:
    """1.0 iff the last <answer>...</answer> uses each input number exactly once
    and evaluates to `target`; 0.0 otherwise."""
    all_matches = re.findall(r"<answer>(.*?)<\/answer>", response, re.DOTALL)
    if not all_matches:
        return 0.0

    answer_content = all_matches[-1].strip()
    if not answer_content:
        return 0.0
    if not re.match(r"^[0-9+\-*/() ]+$", answer_content):
        return 0.0

    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def _coerce_number(x: Union[int, float, str]) -> Union[int, float]:
    """Parse a target value into a number, preserving non-integer values.

    Some countdown targets are non-integers (e.g. "27.4", "266.666..."), so we
    must not force `int()`. We keep integral values as `int` for clean logging
    and fall back to `float` otherwise; downstream scoring compares with
    `float(target)` either way.
    """
    if isinstance(x, (int, float)):
        return x
    f = float(str(x).strip())
    return int(f) if f.is_integer() else f


def _unpack_target(gt_answer: Union[Dict[str, Any], Tuple[List[int], Union[int, str]]]
                   ) -> Tuple[List[int], Union[int, float]]:
    if isinstance(gt_answer, dict):
        numbers, target_value = gt_answer["numbers"], gt_answer["target"]
    else:
        numbers, target_value = gt_answer  # (numbers, target_value)
    return list(numbers), _coerce_number(target_value)


def countdown_reward_fn(
    model_response: str,
    gt_answer: Union[Dict[str, Any], Tuple[List[int], Union[int, str]]],
) -> Tuple[Dict[str, Any], float]:
    """Reward function for the Countdown task.

    The trainer calls ``reward_fn(response_text, target)`` where ``target`` is a
    single item produced by your dataset's ``collate_fn``. Pack both the input
    numbers and the desired result value into that item, e.g.::

        def countdown_collate_fn(batch):
            prompts  = [item["context"] for item in batch]
            targets  = [
                {"numbers": item["numbers"], "target": item["target"]}
                for item in batch
            ]
            return prompts, targets

    Total reward = 0.1 * format_reward + answer_reward (matches the original
    countdown_task.py weighting).
    """
    numbers, target_value = _unpack_target(gt_answer)

    # The countdown prompt ends with "<think>", which vLLM does not echo back —
    # prepend it before format scoring so the regex sees a complete envelope.
    format_reward = format_reward_function("<think>" + model_response)
    answer_reward = answer_reward_function(model_response, numbers, target_value)

    fmt = {
        "formatted": format_reward > 0,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
    return fmt, 0.1 * format_reward + answer_reward
