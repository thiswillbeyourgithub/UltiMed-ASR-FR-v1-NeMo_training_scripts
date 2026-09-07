# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for AdamSPD (Adam with Selective Projection Decay).

The regression these guard against: the SPD criterion carried an inverted
sign, which selected the "step moves toward pretrained" case. That is also
the case where the ratio term is negative and gets hardtanh-clamped to zero,
so the two mistakes cancelled and SPD silently applied no decay at all. The
optimizer behaved as plain Adam with weight_decay ignored entirely, which is
invisible unless you compare trajectories across weight_decay values.

Written with Claude Code.
"""

import pytest
import torch

from nemo.collections.asr.optim.adam_spd import AdamSPD, attach_pretrained_params


def _drift_after_steps(weight_decay, steps=200, lr=2e-5, n=256, seed=0, away=True):
    """Run a fixed synthetic trajectory and return the final distance from pretrained.

    The parameter starts exactly at its pretrained value (as it does in a real
    fine-tune) and receives a constant gradient, so it walks steadily away from
    that value and SPD has something to pull back against.
    """
    torch.manual_seed(seed)
    param = torch.nn.Parameter(torch.randn(n))
    pretrained = param.detach().clone()

    optimizer = AdamSPD([param], lr=lr, weight_decay=weight_decay)
    optimizer.set_pretrained_params_by_id({id(param): pretrained})

    grad = torch.ones(n) * (-1.0 if away else 1.0)
    for _ in range(steps):
        param.grad = grad.clone()
        optimizer.step()

    return (param.detach() - pretrained).norm().item()


class TestAdamSPD:
    def test_weight_decay_actually_does_something(self):
        """The regression test proper: decay must change the trajectory.

        With the inverted sign this failed loudly, every weight_decay produced
        a bit-identical result.
        """
        no_decay = _drift_after_steps(weight_decay=0.0)
        decayed = _drift_after_steps(weight_decay=0.01)

        assert decayed != no_decay, (
            "SPD applied no decay at all: weight_decay=0.01 gave a bit-identical "
            "trajectory to weight_decay=0.0. The selection criterion is inverted."
        )
        assert decayed < no_decay, "SPD decay must pull the parameter back toward pretrained"

    def test_decay_strength_is_monotone(self):
        """Stronger decay must hold the parameter closer to its pretrained value."""
        drifts = [_drift_after_steps(weight_decay=wd) for wd in (0.0, 0.01, 0.1, 0.5)]
        # STRICTLY decreasing on purpose: a plain sorted() check also accepts a
        # constant list, which is exactly the failure mode the inverted sign
        # produced (every weight_decay giving an identical trajectory).
        assert all(a > b for a, b in zip(drifts, drifts[1:])), (
            f"drift must shrink strictly as weight_decay grows, got {drifts}"
        )

    def test_no_decay_setting_matches_plain_adam_path(self):
        """weight_decay=0 must be a no-op regardless of the pretrained snapshot."""
        with_snapshot = _drift_after_steps(weight_decay=0.0)

        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(256))
        optimizer = AdamSPD([param], lr=2e-5, weight_decay=0.0)  # no snapshot registered
        start = param.detach().clone()
        grad = torch.ones(256) * -1.0
        for _ in range(200):
            param.grad = grad.clone()
            optimizer.step()
        without_snapshot = (param.detach() - start).norm().item()

        assert with_snapshot == pytest.approx(without_snapshot, rel=1e-6)

    def test_no_decay_when_step_moves_toward_pretrained(self):
        """The "selective" half of SPD: a step already heading home is left alone.

        Starts the parameter displaced from pretrained and feeds it a gradient
        whose step walks it back, so the criterion must decline to add decay and
        the trajectory must be identical to weight_decay=0.
        """

        def run(weight_decay):
            torch.manual_seed(0)
            pretrained = torch.randn(256)
            param = torch.nn.Parameter(pretrained + 0.5)  # start displaced
            optimizer = AdamSPD([param], lr=2e-5, weight_decay=weight_decay)
            optimizer.set_pretrained_params_by_id({id(param): pretrained.clone()})
            grad = torch.ones(256)  # step is -grad, i.e. back toward pretrained
            for _ in range(200):
                param.grad = grad.clone()
                optimizer.step()
            return (param.detach() - pretrained).norm().item()

        assert run(0.1) == pytest.approx(run(0.0), rel=1e-9)

    def test_attach_pretrained_params_snapshots_only_trainable(self):
        """Frozen parameters must not get a pretrained reference or be stepped."""
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
        for p in model[0].parameters():
            p.requires_grad_(False)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamSPD(trainable, lr=1e-4, weight_decay=0.01)
        attach_pretrained_params(optimizer, model)

        registered = sum(len(g['pre']) for g in optimizer.param_groups)
        assert registered == len(trainable) == 2

    def test_snapshot_survives_state_dict_roundtrip(self):
        """A crash relaunch restores the optimizer, so the snapshot must survive.

        The launcher relaunches on crash and resume_if_exists picks the run back
        up, so losing the reference here would silently disable SPD for the rest
        of a multi-day run.
        """
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(32))
        pretrained = param.detach().clone()
        optimizer = AdamSPD([param], lr=1e-4, weight_decay=0.1)
        optimizer.set_pretrained_params_by_id({id(param): pretrained})

        param.grad = torch.ones(32)
        optimizer.step()
        state = optimizer.state_dict()

        restored = AdamSPD([param], lr=1e-4, weight_decay=0.1)
        restored.load_state_dict(state)

        assert 'pre' in restored.param_groups[0], "pretrained snapshot lost across state_dict roundtrip"
        assert torch.equal(restored.param_groups[0]['pre'][0], pretrained)

    def test_rejects_sparse_gradients(self):
        param = torch.nn.Parameter(torch.randn(4))
        optimizer = AdamSPD([param], lr=1e-4)
        param.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (4,))
        with pytest.raises(RuntimeError, match="does not support sparse gradients"):
            optimizer.step()
