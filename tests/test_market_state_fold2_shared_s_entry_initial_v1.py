import torch

from gx1.audit.qualify_market_state_fold2_shared_s_entry_initial_v1 import (
    common_target_probe,
)


def test_common_mask_probe_excludes_unknown_path_cells():
    net = torch.tensor([[[2., 3.], [99., 99.]]], requires_grad=True)
    mae = torch.tensor([[[1., 2.], [99., 99.]]], requires_grad=True)
    target_net = torch.tensor([[[0., 0.], [float('nan'), float('nan')]]])
    target_mae = torch.tensor([[[0., 0.], [float('nan'), float('nan')]]])
    loss = common_target_probe(
        net, mae, target_net, target_mae,
        torch.tensor([[True, False]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(net.grad[:, 0]) == 2
    assert torch.count_nonzero(mae.grad[:, 0]) == 2
    assert torch.count_nonzero(net.grad[:, 1]) == 0
    assert torch.count_nonzero(mae.grad[:, 1]) == 0
