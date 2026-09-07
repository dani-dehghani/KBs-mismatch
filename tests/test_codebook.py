import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_drift.codebook import (
    build_class_prototypes,
    clone_aligned_codebooks,
    evaluate_aligned_codebooks,
    measure_codebook_drift,
)


class ToyEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(2))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_aligned_class_prototypes_are_independent_and_decode_correctly() -> None:
    inputs = torch.tensor([[2.0, 0.0], [4.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    targets = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = ToyEmbeddingModel()

    canonical, counts = build_class_prototypes(model, loader, 2, torch.device("cpu"))
    transmitter, receiver = clone_aligned_codebooks(canonical)
    drift = measure_codebook_drift(transmitter, receiver)
    evaluation = evaluate_aligned_codebooks(
        model,
        loader,
        transmitter,
        receiver,
        torch.device("cpu"),
    )

    assert torch.equal(canonical, torch.eye(2))
    assert counts.tolist() == [2, 2]
    assert torch.equal(transmitter, receiver)
    assert transmitter.untyped_storage().data_ptr() != receiver.untyped_storage().data_ptr()
    assert drift.mean_cosine_distance == 0.0
    assert drift.mean_euclidean_distance == 0.0
    assert drift.max_absolute_difference == 0.0
    assert evaluation.classifier_accuracy == 1.0
    assert evaluation.transmitter_accuracy == 1.0
    assert evaluation.receiver_accuracy == 1.0
    assert evaluation.transmitter_receiver_agreement == 1.0
