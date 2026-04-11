import torch
import torch.nn as nn
import brevitas.nn as qnn

INPUT_FEATURES = 41
OUTPUT_CLASSES = 2
DEFAULT_HIDDEN_DIMS = (128, 256, 128)


def build_model(bits, hidden_dims=DEFAULT_HIDDEN_DIMS):
    layers = []
    in_features = INPUT_FEATURES

    for out_features in hidden_dims:
        layers.append(
            qnn.QuantLinear(
                in_features,
                out_features,
                bias=True,
                weight_bit_width=bits,
            )
        )
        layers.append(qnn.QuantReLU(bit_width=bits))
        in_features = out_features

    layers.append(
        qnn.QuantLinear(
            in_features,
            OUTPUT_CLASSES,
            bias=True,
            weight_bit_width=bits,
        )
    )
    return nn.Sequential(*layers)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for bits in [2, 4, 8]:
        model = build_model(bits)
        total_params = parameter_count(model)
        memory_kb = (total_params * bits) / (8 * 1024)
        print(
            f"{bits}-bit │ hidden={DEFAULT_HIDDEN_DIMS} │ "
            f"parameters: {total_params} │ memory: {memory_kb:.2f} KB"
        )
