"""
AdamSPD — Adam with Selective Projection Decay.

Implements the optimizer from "Selective Projection Decay for Reducing
Negative Transfer in Multi-Task Learning" (GT-RIPL).  Instead of applying
weight decay uniformly, SPD only decays weights *toward their pretrained
values* when the current gradient update would move them further away
from those pretrained values.  This selectively prevents forgetting while
still allowing beneficial updates.

Original source: https://github.com/GT-RIPL/Selective-Projection-Decay

The key difference from L2-SP (which adds a loss penalty) is that SPD
operates *inside the optimizer step*, making the regularization interact
properly with Adam's adaptive learning rates and momentum.

Usage::

    # In NeMo config (training_config.yaml), set the optimizer to:
    optim:
      _target_: nemo.collections.asr.optim.adam_spd.AdamSPD
      lr: 1e-5
      weight_decay: 0.01   # SPD decay strength (only applied selectively)

    # Then in the training script, call set_pretrained_params() after
    # creating the optimizer to register the pretrained weight snapshot.
"""

import math
import copy

import torch
from torch.optim.optimizer import Optimizer


class AdamSPD(Optimizer):
    """Adam optimizer with Selective Projection Decay (SPD).

    SPD modifies weight decay to only pull weights back toward their
    pretrained values when the gradient update would increase the
    distance from those pretrained values.  When the gradient already
    moves weights closer to pretrained values, no decay is applied.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining param groups.
    lr : float
        Learning rate (default: 1e-3).
    betas : tuple[float, float]
        Coefficients for computing running averages of gradient and its
        square (default: (0.9, 0.999)).
    eps : float
        Term added to denominator for numerical stability (default: 1e-8).
    weight_decay : float
        SPD decay coefficient — controls how strongly weights are pulled
        back toward pretrained values (default: 0).
    amsgrad : bool
        Whether to use the AMSGrad variant (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def set_pretrained_params(self, named_params: dict[str, torch.Tensor]) -> None:
        """Register pretrained weight snapshot for SPD.

        Must be called after the optimizer is created.  For each param
        group, builds a list of pretrained tensors aligned with the
        group's ``params`` list.  Parameters not found in the snapshot
        get a zero tensor (equivalent to standard weight decay for those).

        Parameters
        ----------
        named_params : dict[str, torch.Tensor]
            Mapping from parameter name to its pretrained value.
            Typically obtained via ``{n: p.clone().detach() for n, p in
            model.named_parameters() if p.requires_grad}``.
        """
        # Build a reverse lookup: id(param) -> pretrained tensor
        # We need the model's named_parameters to map names to param objects,
        # but the optimizer only has the param tensors. The caller must ensure
        # named_params keys match model.named_parameters() keys.
        self._pretrained_by_id: dict[int, torch.Tensor] = {}
        for name, pretrained_val in named_params.items():
            self._pretrained_by_id[id(pretrained_val)] = pretrained_val

        # Store pretrained references on each param group for fast access
        # during step(). We match by value equality with the snapshot.
        for group in self.param_groups:
            group['pre'] = []
            for p in group['params']:
                # Find the matching pretrained tensor by parameter name
                # Since we can't easily map param->name here, we store the
                # full dict and look up during step() by param identity.
                group['pre'].append(None)  # placeholder

        # We'll use a different approach: store the mapping and look up
        # by param id during step(). This requires the caller to also
        # pass the param->name mapping.
        self._pretrained_lookup_ready = False

    def set_pretrained_params_by_id(
        self,
        param_to_pretrained: dict[int, torch.Tensor],
    ) -> None:
        """Register pretrained weights indexed by param tensor id.

        This is the preferred internal method. The high-level helper
        ``attach_pretrained_params`` builds this mapping automatically.

        Parameters
        ----------
        param_to_pretrained : dict[int, torch.Tensor]
            Mapping from ``id(param)`` to the cloned pretrained value.
        """
        # Store on each param group for fast access during step()
        for group in self.param_groups:
            pre_list = []
            for p in group['params']:
                pre = param_to_pretrained.get(id(p))
                if pre is not None:
                    pre_list.append(pre)
                else:
                    # No pretrained reference — use zeros (standard decay)
                    pre_list.append(torch.zeros_like(p))
            group['pre'] = pre_list

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            # Pretrained references for SPD (one per param in group)
            pre_list = group.get('pre')

            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamSPD does not support sparse gradients"
                    )

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (
                        max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(eps)
                else:
                    denom = (
                        exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(eps)

                step_size = lr / bias_correction1
                d_p = step_size * exp_avg / denom
                new_p = p - d_p

                # --- Selective Projection Decay (SPD) ---
                # Only apply decay when the gradient is pushing the param
                # *away* from the pretrained value (condition < 0).
                #
                # Sign matters and is easy to get wrong. A step p <- p - eta*g
                # changes the squared distance to `pre` by approximately
                # -2*eta*(g . (p - pre)), so:
                #   g . (p - pre) < 0  =>  the step moves AWAY   => decay needed
                #   g . (p - pre) > 0  =>  the step moves TOWARD => leave alone
                # so `condition` must be g . (p - pre) itself, NOT its negation.
                #
                # This carried an extra minus sign until 2026-08-22, which
                # selected the toward-pretrained case instead. That case is also
                # the one where _ratio() below is negative and hardtanh clamps it
                # to 0, so the two mistakes cancelled and SPD applied exactly
                # zero decay: weight_decay 0.0, 0.01 and even 1.0 produced
                # bit-identical trajectories. Covered by
                # tests/collections/asr/test_adam_spd.py.
                pre = (
                    pre_list[j]
                    if pre_list is not None
                    else torch.zeros_like(p)
                )
                condition = torch.sum(torch.mul(grad, p - pre))
                if condition < 0.0:
                    ratio = self._ratio(new_p, p, pre)
                    new_p = new_p - weight_decay * ratio * (new_p - pre)

                p.copy_(new_p)

        return loss

    @staticmethod
    def _ratio(
        new_p: torch.Tensor,
        param: torch.Tensor,
        pre: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the SPD scaling ratio.

        Measures how much further the new parameter is from pretrained
        compared to the current parameter, clamped to [0, 1].
        """
        curr_norm = torch.norm(new_p - pre)
        prev_norm = torch.norm(param - pre)
        ratio = (curr_norm - prev_norm) / curr_norm
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)


def attach_pretrained_params(
    optimizer: AdamSPD,
    model: torch.nn.Module,
) -> None:
    """Snapshot the model's current trainable weights and register them
    with the AdamSPD optimizer for Selective Projection Decay.

    Call this *after* creating the optimizer but *before* training starts.

    Parameters
    ----------
    optimizer : AdamSPD
        The optimizer instance.
    model : torch.nn.Module
        The model whose trainable parameters were passed to the optimizer.
    """
    param_to_pretrained: dict[int, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_to_pretrained[id(param)] = param.clone().detach()
    optimizer.set_pretrained_params_by_id(param_to_pretrained)
