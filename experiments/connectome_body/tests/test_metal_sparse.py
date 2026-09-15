import numpy as np
import pytest
import torch
from scipy import sparse

from connectome_body.adaptation.imitation import (
    AdapterSpec,
    Optimization,
    load_actor,
    train_offline,
)
from connectome_body.adaptation.models import GenericAdapter
from connectome_body.adaptation.temporal import temporal_cache
from connectome_body.graphs import make_fixture
from connectome_body.metal_sparse import FrozenMetalCSR
from connectome_body.util import seed_everything

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable"
)


@pytest.mark.parametrize("n,batch", [(7, 1), (35, 3), (128, 4)])
def test_metal_signed_directed_csr_matches_cpu_values_and_gradients(n, batch):
    rng = np.random.default_rng(n)
    dense = rng.normal(size=(n, n)).astype(np.float32)
    dense[rng.random((n, n)) < 0.7] = 0
    dense[0] = 0  # Empty row and a non-tile-aligned graph size.
    matrix = sparse.csr_matrix(dense)
    metal = FrozenMetalCSR(matrix).to("mps")
    cpu = torch.tensor(rng.normal(size=(n, batch)).astype(np.float32)).T.requires_grad_()
    gpu = cpu.detach().to("mps").requires_grad_()
    expected = cpu @ torch.from_numpy(dense).T
    actual = metal(gpu)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(gpu.grad.cpu(), cpu.grad, rtol=2e-5, atol=3e-6)
    assert not list(metal.parameters())


def test_metal_learned_ports_and_offline_resume(tmp_path):
    graph = make_fixture(tmp_path / "graph", 64, 0)
    seed_everything(0)
    actor = GenericAdapter(3, 1, 1000, 4, 8, graph, backend="mps").to("mps")
    context, state = actor.context(), actor.reset(2)
    for _ in range(8):
        prediction, state = actor(torch.ones(2, 3, device="mps"), state, context)
    prediction.square().sum().backward()
    for name in ("input_queries", "output_queries"):
        grad = getattr(actor.ports, name).grad
        assert grad.device.type == "mps" and torch.isfinite(grad).all() and grad.norm() > 0
    train = temporal_cache(tmp_path / "train", episodes=4, length=8)
    validation = temporal_cache(tmp_path / "val", episodes=4, length=8, split="validation")
    spec = AdapterSpec(
        graph=str(tmp_path / "graph"), budget=1000, channels=4, support=8, device="mps"
    )
    opt = Optimization(
        updates=2, batch_size=2, sequence_length=4, burn_in=4, eval_every=1, checkpoint_every=1
    )
    train_offline(spec, opt, [train.path], validation.path, tmp_path / "full")
    train_offline(spec, opt, [train.path], validation.path, tmp_path / "resume", stop_after=1)
    train_offline(spec, opt, [train.path], validation.path, tmp_path / "resume", resume=True)
    first, _, _ = load_actor(tmp_path / "full")
    second, _, _ = load_actor(tmp_path / "resume")
    for name, value in first.state_dict().items():
        torch.testing.assert_close(
            value.cpu(), second.state_dict()[name].cpu(), rtol=2e-5, atol=2e-6
        )
