"""GPU: the KL-to-base penalty is 0 for an untrained adapter, positive and differentiable once the adapter moves,
and only acts on replay rows."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU 4-bit model")


def test_kl_zero_at_init_positive_after_update_and_replay_only():
    from sleepllm import sleep as S
    from sleepllm.config import Config
    from sleepllm.llm import load_model
    cfg = Config()
    cfg.lora.dropout = 0.0
    tok, model = load_model(cfg.model.name)
    model = S.attach_adapter(model, cfg.lora)
    model.train()
    samples = [{"user": "Where does Mina Tor live?", "assistant": "Dova"},
               {"user": "Continue the text: The river runs", "assistant": "south to the sea past the old harbour town."}]
    enc = [S._encode(tok, s, 256) for s in samples]
    ids, lab, att = S._batch(tok, enc)
    ntok = int((lab[:, 1:] != -100).sum())
    rep = torch.tensor([False, True], device=ids.device)
    with torch.no_grad():
        ce = S.batch_loss(model, ids, lab, att, ntok, True).item()
        at_init = S.batch_loss(model, ids, lab, att, ntok, True, rep, 1.0).item()
    assert abs(at_init - ce) < 2e-3, (at_init, ce)                        # LoRA B = 0 -> adapter == base -> KL ~ 0
    for n, p in model.named_parameters():
        if p.requires_grad and "lora_B" in n:
            p.data.normal_(0, 0.02)
    with torch.no_grad():
        ce2 = S.batch_loss(model, ids, lab, att, ntok, True).item()
        with_kl = S.batch_loss(model, ids, lab, att, ntok, True, rep, 1.0).item()
        none_rep = S.batch_loss(model, ids, lab, att, ntok, True, torch.zeros_like(rep), 1.0).item()
    assert with_kl > ce2 + 1e-4                                            # replay row now contributes KL
    assert abs(none_rep - ce2) < 1e-6                                      # no replay rows -> no penalty
    model.zero_grad()
    S.batch_loss(model, ids, lab, att, ntok, True, rep, 1.0).backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
    assert g > 0
