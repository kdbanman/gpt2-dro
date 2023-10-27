# gpt2-dro

This repository attempts to replicate the GPT-2 training process using a Conditional Value-at-Risk Distributionally Robust Optimization (CVaR DRO) optimality target, rather than the common Expected Risk Minimization (ERM) optimality target.

This is achieved with [a performant minibatch ɑ-CVaR DRO](https://arxiv.org/abs/2010.05893) implementation.

Since the GPT-2 training dataset (WebTex) is proprietary to OpenAI, the [OpenWebText dataset](https://paperswithcode.com/dataset/openwebtext) is used instead.

The primary file is `gpt2-openwebtext-dro.py`, containing data loading, training, and logging code.
- HuggingFace is used for model checkpoints.

  For example, [these publicly accessible models](https://huggingface.co/kdbanman?search_models=gpt2-openwebtext) trained from this repo.  ⍺ refers to the uncertainty set size for ⍺-CVaR DRO

  <kbd><img width="696" alt="kdbanman__Kirby_Banman_" src="https://github.com/kdbanman/gpt2-dro/assets/1323521/5576a913-4ea7-4e53-8226-a5224357947d"></kbd>



- WandB is used for training logs.

  For example, these monitoring ~60 and ~200 hour training runs for the models above.  Again, ⍺ refers to the uncertainty set size for ⍺-CVaR DRO

  <kbd><img width="819" alt="gpt2-dro_Workspace_–_Weights___Biases" src="https://github.com/kdbanman/gpt2-dro/assets/1323521/01655e71-7592-418d-82b8-1545c778a0af"></kdb>


The training is designed for data parallelism across multiple large<sup>*</sup> GPUs within a single node using the [HuggingFace Accelerate library](https://huggingface.co/docs/accelerate/index).  For example,

```bash
accelerate launch gpt2-openwebtext-dro.py config-0.8.json
```

which runs the experiment configuration in `config-0.8.json`.  ^This command is exactly what the `launch-0.8.sh` file does headlessly directly on a compute node, and what the `slurm_launch.sh` file does from a login node in a slurm cluster.

------

<sup>*</sup> "large" GPUs as of early 2023 - the models and gradients fit fine in ~20GB VRAM with modest batch sizes (less than 100 samples)
