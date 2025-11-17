import torch

from astrafier.models import AstrafierModule, ClassificationHead, LightCurveEncoder


def _make_batch(batch: int = 2, seq_len: int = 32):
    torch.manual_seed(0)
    flux = torch.randn(batch, seq_len)
    time = torch.linspace(0, 1, seq_len).repeat(batch, 1)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    labels = torch.nn.functional.one_hot(
        torch.randint(0, 8, (batch,)), num_classes=8
    ).to(torch.float32)
    return flux, time, labels, mask


def test_light_curve_classifier_embeddings():
    model = LightCurveEncoder()
    flux, time, _, mask = _make_batch()

    embeddings = model(flux, time, mask)
    assert embeddings.shape == (flux.shape[0], 64)


def test_head_only_forward():
    head = ClassificationHead()
    features = torch.randn(4, 64)

    logits = head(features)
    assert logits.shape == (4, 8)


def test_combined_forward_backward_and_predict():
    model = AstrafierModule(class_weight=torch.ones(8))
    flux, time, labels, mask = _make_batch()

    logits = model(flux, time, mask)
    assert logits.shape == (flux.shape[0], 8)

    loss = torch.nn.functional.cross_entropy(logits, labels.argmax(dim=1))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    ticids = torch.arange(flux.shape[0], dtype=torch.int32)
    preds = model.predict_step((flux, time, ticids, mask))
    assert isinstance(preds, list)
    assert len(preds) == flux.shape[0]
    assert "tic" in preds[0] and "probabilities" in preds[0]
    assert len(preds[0]["probabilities"]) == 8

