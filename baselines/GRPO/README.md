# GRPO:Zero

GRPO training with minimal dependencies (and low GPU memory usage!). We implement almost everything from scratch and only depend on `tokenizers` for tokenization and `pytorch` for training. 
- No `transformers` and `vLLM` dependencies! 
- The default config is set to run on a single A40 GPU (48GB VRAM) for a few hours to get good results. (An A40 costs `$0.44` per hour if you rent it from RunPod.)
- We also support training with a 24GB VRAM GPU (e.g., an RTX 4090 GPU) by offloading the optimizer to CPU. Fortunately, this only adds a small overhead to the training because we only update the policy network a few hundred times during the entire training process.
- We support several improvements over the original GRPO algorithm from the [DAPO project](https://arxiv.org/abs/2503.14476), including:
    - **Token-level policy gradient loss**: every token is equally weighted in the policy gradient loss.
    - **Removing KL Divergence**: the KL divergence is not used in the policy gradient loss. This reduces GPU memory usage as we no longer need the reference policy network.
    - **Overlong episode filtering**: skips unfinished episodes that exceed context length limits. This stabilizes training. Though we disabled it by default to observe model learning under limited context length. Set `skip_unfinished_episodes` to `true` to enable it.

## Algorithm 

Group Relative Policy Optimization (GRPO) is an algorithm proposed by Deepseek for training large language models with reinforcement learning. The idea is simple: for each question, we randomly sample multiple answers. The advantage of an answer is then defined as the normalized reward. This gets rid of the value estimation network. In particular, we implement the following algorithm:

1. For each training step, randomly sample $N$ questions $q_1, q_2, \cdots, q_N$.
2. For each question $q_i$, sample $M$ answers $a_{i,1}, a_{i,2}, \cdots, a_{i,M}$.
3. Compute the reward $r_{i,j}$ for each answer $a_{i,j}$.
4. Compute the mean and std of the rewards for each question $q_i$.

$$
\begin{aligned}
\mu_i &\leftarrow \text{mean}(r_{i,1}, r_{i,2}, \cdots, r_{i,M}) \\
\sigma_i &\leftarrow \text{std}(r_{i,1}, r_{i,2}, \cdots, r_{i,M})
\end{aligned}
$$

5. For each token $t$ in the answer $a_{i,j}$, compute the advantage as

$$A_{i,j}[t] \leftarrow \frac{r_{i,j} - \mu_i}{\sigma_i}$$

6. Compute policy gradient using PPO surrogate objective. For simplicity, we will only do one policy update per iteration, in which the gradient of the PPO objective is equivalent to following vanilla policy gradient estimation (per token).

$$
\nabla_\theta \log \pi_\theta(a_{i,j}[t]) \cdot A_{i,j}[t]
$$

7. Update the policy network $\pi(\theta)$ using the gradient. Go back to step 1.

## CountDown Task

We are going to train the Qwen2.5 models on the [CountDown task](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4). Given a list of 3 or 4 numbers and a target number, the model needs to generate a mathematical expression using simple arithmetic operations (+, -, *, /) that evaluates to the target number. For example:

```
Question: Given 1 2 3 4 and a target number 11. Show an expression that evaluates to 11.
Answer: 1 + (2 * 3) + 4
```

## Reward Function

To solve the CountDown task, we will use the GRPO algorithm to train the model to generate the chain of thought reasoning before generating the final expression. Specifically, the model is trained to follow the format:

```
<think>Model step by step reasoning</think>
<answer>Final answer</answer>
```

The reward is the sum of two components:

1. **Format Reward**: The model earns a reward of `0.1` when it correctly follows the specified format with thinking and answer tags, and `0` otherwise.
2. **Answer Reward**: The model receives a reward of `1` if its final answer uses each provided number exactly once and correctly evaluates to the target value, otherwise it receives `0`.


## Training

We use the `Qwen2.5-3B-Instruct` model for training. To train the model, run the following commands:

```bash
# initialize the environment
pip install uv
uv sync

# install git-lfs
apt update; apt install git-lfs -y; git lfs install

# download the dataset
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

# download the pretrained model
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# train the model
uv run train.py
# train the model with a 24GB VRAM GPU (e.g., an RTX 4090 GPU)
uv run train.py --config config_24GB.yaml
```
## Acknowledgements

This project builds upon the work of several outstanding projects:

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) for pioneering the GRPO algorithm.
- [DAPO](https://arxiv.org/abs/2503.14476) for their enhancements to the original GRPO algorithm.
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) for their implementation of GRPO and creation of the [CountDown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) dataset.
- [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment/tree/main) for their clear implementation and tutorial on the GRPO algorithm.
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) for developing the high-quality pretrained model used in this project.