import torch
import torch.nn.functional as F
from torch import nn


import math
from typing import Literal, Optional


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        bias: bool,
        init_lora_weight: Optional[Literal["default", "pissa"]] = None,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.rank = rank
        self.alpha = alpha
        self.scaled_alpha = alpha / math.sqrt(rank) if use_rslora else alpha / rank
        self.merged = False

        self.lora_a = nn.Parameter(
            torch.empty(
                (rank, in_features),
                device=self.linear.weight.device,
            )
        )
        self.lora_b = nn.Parameter(
            torch.empty(
                (out_features, rank),
                device=self.linear.weight.device,
            )
        )

        self.reset_lora_parameters(init_lora_weight)

    def reset_lora_parameters(
        self, init_lora_weight: Literal["default", "pissa"]
    ) -> None:
        if init_lora_weight == "default":
            nn.init.kaiming_normal_(self.lora_a)
            nn.init.zeros_(self.lora_b)
        elif init_lora_weight == "pissa":
            Vr, Sr, Ur = torch.svd_lowrank(
                self.linear.weight.data.to(torch.float32), q=self.rank
            )
            Sr /= self.alpha / self.rank
            Uhr = Ur.t()

            self.lora_a.data = torch.diag(torch.sqrt(Sr)) @ Uhr
            self.lora_b.data = Vr @ torch.diag(torch.sqrt(Sr))
            self.linear.weight.data -= (
                (self.scaled_alpha) * self.lora_b @ self.lora_a
            ).to(self.linear.weight.dtype)

    def merge(self) -> None:
        if self.merged:
            return

        lora_dtype = self.lora_a.dtype
        dtype = self.linear.weight.dtype

        with torch.no_grad():
            self.linear.weight.data = (
                self.linear.weight.data.to(lora_dtype)
                + (self.alpha / self.rank) * self.lora_b @ self.lora_a
            ).to(dtype)

        self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)

        if self.merged:
            return output

        lora_output = F.linear(x.to(self.lora_a.dtype), self.lora_a)
        lora_output = F.linear(lora_output, self.lora_b)
        return (output + (self.scaled_alpha) * lora_output).to(self.linear.weight.dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_linear: nn.Linear,
        rank: int,
        alpha: int,
        init_lora_weight: Literal["default", "pissa"],
    ) -> "LoRALinear":
        with torch.device("meta"):
            lora_linear = cls(
                pretrained_linear.in_features,
                pretrained_linear.out_features,
                rank,
                alpha,
                pretrained_linear.bias is not None,
            )
        lora_linear.to_empty(device=pretrained_linear.weight.device)
        lora_linear.linear.weight = pretrained_linear.weight
        if pretrained_linear.bias is not None:
            lora_linear.linear.bias = pretrained_linear.bias

        lora_linear.reset_lora_parameters(init_lora_weight)
        return lora_linear
