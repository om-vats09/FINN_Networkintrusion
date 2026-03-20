import torch
import torch.nn as nn
import brevitas.nn as qnn

def build_model(bits):
    return nn.Sequential(
        qnn.QuantLinear(41, 64,  bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(64, 128, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(128, 64, bias=True, weight_bit_width=bits),
        qnn.QuantReLU(bit_width=bits),
        qnn.QuantLinear(64, 2,   bias=True, weight_bit_width=bits),
    )

if __name__ == "__main__":
    for bits in [2, 4, 8]:
        model = build_model(bits)
        total_params = sum(p.numel() for p in model.parameters())
        memory_kb    = (total_params * bits) / (8 * 1024)
        print(f"{bits}-bit │ parameters: {total_params} │ memory: {memory_kb:.2f} KB")