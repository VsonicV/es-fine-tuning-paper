

def qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def countdown_template(question: str):
    # The countdown dataset's `context` field already contains the system
    # instructions, the formatted question, and the opening `<think>` tag —
    # the trainer should feed that text straight through to the model.
    return question