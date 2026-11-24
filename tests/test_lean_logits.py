"""The memory-lean loss must equal the full-logits loss (value and LoRA gradients)."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU 4-bit model")


def test_lean_loss_and_grads_match_full():
    from sleepllm import sleep as S
    from sleepllm.config import Config
    from sleepllm.llm import load_model
    cfg = Config()
    cfg.lora.dropout = 0.0
    tok, model = load_model(cfg.model.name)
    model = S.attach_adapter(model, cfg.lora)
    model.train()
    samples = [{"user": "Where does Mina Tor live?", "assistant": "Dova"},
               {"user": "Name a fact about Karst Labs.", "assistant": "Karst Labs is headquartered in Wenne, near the river."},
               {"user": "Q?", "assistant": "A"}]
    enc = [S._encode(tok, s, 256) for s in samples]
    ids, lab, att = S._batch(tok, enc)
    ntok = int((lab[:, 1:] != -100).sum())
    params = [p for p in model.parameters() if p.requires_grad]

    def run(lean):
        model.zero_grad()
        loss = S.batch_loss(model, ids, lab, att, ntok, lean)
        loss.backward()
        g = torch.cat([p.grad.flatten().float() for p in params if p.grad is not None])
        return loss.item(), g

    l_full, g_full = run(False)
    l_lean, g_lean = run(True)
    assert abs(l_full - l_lean) / abs(l_full) < 5e-3
    cos = torch.nn.functional.cosine_similarity(g_full, g_lean, dim=0).item()
    assert cos > 0.99, cos
