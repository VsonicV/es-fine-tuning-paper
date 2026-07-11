from setuptools import find_packages, setup

# Core is intentionally backend-agnostic and vLLM-free so it can be installed
# into the SGLang worker's conda env (which must not pull in vllm). Backend
# engines come from extras:
#   pip install -e ".[vllm]"     # trainer + vLLM engines (es-prefix-cache env)
#   pip install -e ".[sglang]"   # SGLang worker (sglang-es env); sglang installed separately
required = [
    "numpy",
    "torch>=2.0.0",
    "transformers",  # exact version is a per-backend concern (see extras) -- do NOT pin in core:
                     # vLLM 0.11 wants 4.57.6, SGLang 0.5.6.post2 wants 4.57.1.
    "psutil",
    "datasets",
    "matplotlib>=3.5.0",
]

extras = {
    "vllm": ["vllm==0.11.0", "transformers==4.57.6", "ray", "accelerate>=0.20.0"],
    "sglang": ["pyzmq"],  # the sglang wheel (which pins transformers==4.57.1) is installed separately in its env
    "dev": ["black==22.8.0", "flake8==5.0.4"],
}


setup(
    name="es-at-scale",
    version="0.0.1",
    description="Python code to fine-tune LLMs with Evolution Strategies.",
    author="Cognizant AI Lab",
    packages=find_packages(include=["es_at_scale", "es_at_scale.*"]),
    install_requires=required,
    extras_require=extras,
)
