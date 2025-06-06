import torch
from torch import Tensor

# supostamente pode ser melhor usar essa do que addcdiv e etc
def addcdiv_stochastic_buffered_(input_bf16: Tensor, buffer_fp32: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0):
    """
    Versão otimizada que usa um buffer float32 pré-alocado.
    input_bf16 + (tensor1 / tensor2 * value)
    
    Args:
        input_bf16: O tensor de entrada/saída (bfloat16).
        buffer_fp32: Um buffer pré-alocado com o mesmo shape, mas dtype float32.
        tensor1, tensor2, value: Argumentos para addcdiv.
    """
    # 1. Copia o valor atual de bf16 para o buffer fp32
    buffer_fp32.copy_(input_bf16)
    
    # 2. Realiza a operação em precisão total (float32) no buffer
    buffer_fp32.addcdiv_(tensor1, tensor2, value=value)
    
    # 3. Copia o resultado de volta para o tensor original com arredondamento estocástico
    copy_stochastic_(input_bf16, buffer_fp32)

def copy_stochastic_(target: Tensor, source: Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def add_stochastic_(input: Tensor, other: Tensor, alpha: float = 1.0):
    """
    adds other to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
    """
    result = other.clone() if other.dtype == torch.float32 else other.to(dtype=torch.float32)

    result.add_(input, alpha=alpha)
    copy_stochastic_(input, result)


def addcdiv_stochastic_(input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0):
    """
    adds (tensor1 / tensor2 * value) to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        tensor1: the numerator tensor
        tensor2: the denominator tensor
        value: a multiplier for tensor1/tensor2
    """
    result = input.clone() if input.dtype == torch.float32 else input.to(dtype=torch.float32)

    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(input, result)