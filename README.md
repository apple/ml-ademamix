# AdEMAMix
Pytorch and Optax implementations of the AdEMAMix optimizer. 

Link to paper: https://arxiv.org/abs/2409.03137

## Usage

### Pytorch

See [this link](https://pytorch.org/) to install Pytorch.

AdEMAMix can be used like any other Pytorch optimizer:

```python
    import torch
    from ademamix import AdEMAMix

    # AdEMAMix parameters
    lr = 1e-3
    betas = (0.9, 0.999, 0.9999)
    alpha = 8.0
    beta3_warmup = alpha_warmup = num_iterations = 256_000
    weight_decay = 0.1

    # your prefered model
    model = ...

    # create an AdEMAMix optimizer 
    opt = AdEMAMix(params=model.parameters(), 
                   lr=lr, 
                   betas=betas, 
                   alpha=alpha, 
                   beta3_warmup=beta3_warmup, 
                   alpha_warmup=alpha_warmup, 
                   weight_decay=weight_decay)

    # training loop
    for itr in range(num_iterations):
        x, y = get_batch()
        loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
```

Remark: When `beta_1` is set to zero, the `m1` buffer is not allocated, saving memory.

### Optax

See [this link](https://optax.readthedocs.io/en/latest/) to install Optax.

AdEMAMix can be used as any other Optax optimizer:

```python
    import optax
    import jax
    from ademamix import ademamix, alpha_scheduler, beta3_scheduler

    # AdEMAMix parameters
    lr = 1e-3
    b1, b2, b3 = (0.9, 0.999, 0.9999)
    alpha = 8.0
    b3_warmup = alpha_warmup = num_iterations = 256_000
    weight_decay = 0.1

    # the function you want to optimize
    loss = ...

    # when using alpha or beta schedulers, those are functions 
    # given to the optimizer, else None
    f_a = alpha_scheduler(alpha, alpha_start=0, warmup=alpha_warmup)
    f_b3 = beta3_scheduler(b3, beta_start=b1, warmup=b3_warmup)

    # define the optimizer function
    opt = ademamix(lr=lr, 
                   b1=b1, 
                   b2=b2, 
                   b3=b3, 
                   alpha=alpha, 
                   b3_scheduler=f_b3, 
                   alpha_scheduler=f_a,
                   weight_decay=weight_decay)

    # training loop
    params = init_params(...)
    opt_state = opt.init(params)
    for itr in range(num_iterations):
        x, y = get_batch()
        grad = jax.grad(loss)(params, x, y)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
```

### Citation

```
@misc{pagliardini2024ademamix,
      title={The AdEMAMix Optimizer: Better, Faster, Older}, 
      author={Matteo Pagliardini and Pierre Ablin and David Grangier},
      year={2024},
      eprint={2409.03137},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.03137}, 
}
```
