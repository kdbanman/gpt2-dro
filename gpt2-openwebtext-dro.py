import json, sys

import huggingface_hub as hf_hub
import wandb

from tqdm import tqdm

import torch

from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, get_scheduler


def read_key(filename):
    with open(filename) as f:
        key = f.read().strip()
    return key


def load_config(filename):
    with open(filename) as f:
        config = json.load(f)
    return config


def get_dataloaders(directory_name, batch_size):
    tokenized_datasets = load_from_disk(directory_name)
    
    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

    return train_dataloader, eval_dataloader


def dro_loss(inputs, logits, alpha=0.8):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)

    if alpha < 1.0:
        # Keep only largest alpha-fraction of losses by reweighting smallest (1-alpha)-fraction to zero
        num_samples = len(loss_per_sample)
        num_to_ignore = num_samples - int(num_samples * alpha)

        if num_to_ignore >= 1 or num_to_ignore < num_samples:
            cutoff_value, _cutoff_index = torch.kthvalue(loss_per_sample, num_to_ignore, dim=0)
            loss_per_sample[loss_per_sample < cutoff_value] = 0
        else:
            print("ERROR: crazy reweighting request from the following.  Skipping DRO reweighting.")
            print(f'alpha: {alpha}')
            print(f'num_samples: {num_samples}')
            print(f'num_to_ignore: {num_to_ignore}')
            print(f'losses: {loss_per_sample}')
    
    return loss_per_sample.mean()


def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def evaluate(accelerator, model, eval_dataloader, max_eval_batches=None):
    model.eval()
    losses = []

    if max_eval_batches is None:
        max_eval_batches = len(eval_dataloader)
        
    for step, batch in tqdm(
        enumerate(eval_dataloader), total=max_eval_batches
    ):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
        
        if step >= max_eval_batches:
            break

    if len(losses[0].shape) == 0:
        loss = torch.mean(torch.stack(losses))
    else:
        loss = torch.mean(torch.cat(losses))
        
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def run_experiment(config_filename):
    hf_hub.login(token=read_key('huggingface.key'), write_permission=True, add_to_git_credential=True)
    wandb.login(key=read_key('wandb.key'))

    config = load_config(config_filename)

    train_dataloader, eval_dataloader = get_dataloaders('tokenized-openwebtext', config['batch_size'])

    # For now, the dataset is tokenized in advance using context_length, so this value is fixed
    # at tokenization time.  In the future, tokenization should really be streamed.  Then 
    # context_length can be varied.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel(AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=config['context_length'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    optimizer = torch.optim.AdamW(get_grouped_params(model, config['weight_decay']), lr=config['learning_rate'])

    accelerator = Accelerator(cpu=False, log_with='wandb')
    accelerator.init_trackers(
        project_name=config['project_name'],
        init_kwargs={'wandb': {
            'name': f'⍺={config["cvar_alpha"]:0.2f}',
        }},
        config=config
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = config['num_train_epochs'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config['learning_rate_warmup_steps'],
        num_training_steps=num_training_steps,
    )

    if accelerator.is_main_process:
        repo_name = hf_hub.get_full_repo_name(config['model_name'])
        try:
            hf_hub.model_info(repo_name)
        except hf_hub.utils.RepositoryNotFoundError:
            hf_hub.create_repo(repo_name)
            
        output_dir = config['model_name']
        repo = hf_hub.Repository(output_dir, clone_from=repo_name)

    gradient_steps_since_eval = 0
    gradient_steps = 0

    model.train()
    for epoch in range(1, config['num_train_epochs'] + 1):
        for epoch_step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=len(train_dataloader)
        ):
            # Note: this code was adapted from a tutorial, and does not properly use
            # Accelerate's gradient accumulation system:
            #
            #     https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
            #
            # A significant speedup is possible if we do that, especially on multinode runs.
            
            logits = model(batch["input_ids"]).logits
            loss = dro_loss(batch["input_ids"], logits, alpha=config['cvar_alpha'])
            log_payload = {
                    "lr": lr_scheduler.get_lr(),
                    "samples/contexts": epoch * epoch_step * config['batch_size'],
                    "samples/tokens": epoch * epoch_step * config['batch_size'] * config['context_length'],
                    "gradient_steps": gradient_steps,
                    "loss/train": loss.item() * config['gradient_accumulation_steps'],
                }
            if epoch_step % config['epoch_step_logging_interval'] == 0:
                accelerator.log(log_payload)
            loss = loss / config['gradient_accumulation_steps']
            accelerator.backward(loss)
            if epoch_step % config['gradient_accumulation_steps'] == 0:
                accelerator.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                gradient_steps += 1
                gradient_steps_since_eval += 1
                
            if gradient_steps_since_eval >= config['gradient_step_eval_interval']:
                gradient_steps_since_eval = 0
                eval_loss, perplexity = evaluate(accelerator, model, eval_dataloader, config['num_eval_batches'])
                accelerator.log(log_payload | {"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                    tokenizer.save_pretrained(output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch step {epoch * epoch_step}", blocking=False
                    )
    
    accelerator.end_training()


def main():
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <experiment_config.json>')
        quit(1)
        
    run_experiment(sys.argv[1])


if __name__ == '__main__':
    main()
