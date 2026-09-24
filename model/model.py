from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalAttention


DEFAULT_IGNORE_INDEX = -100


class GPTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = CausalAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        self.ln1 = nn.LayerNorm(
            embed_dim
        )

        self.ln2 = nn.LayerNorm(
            embed_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                embed_dim,
                4 * embed_dim,
            ),
            nn.GELU(),
            nn.Linear(
                4 * embed_dim,
                embed_dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


class GPT(nn.Module):
    """
    GPT-модель NOVA.

    Поддерживает две совместимые архитектуры:

    tie_weights=True
        Новая финальная NOVA.
        token_embedding и LM head используют одну матрицу.

    tie_weights=False
        Старая NOVA v1 и technical_v1_1.
        token_embedding и LM head независимы.

    ВАЖНО:
    Само значение tie_weights не меняет размеры слоёв.
    Оно меняет только совместное использование весов
    embedding/output head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        block_size: int = 256,
        dropout: float = 0.1,
        tie_weights: bool = True,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.block_size = block_size
        self.dropout_p = dropout
        self.tie_weights = tie_weights
        self.ignore_index = ignore_index

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embed_dim,
        )

        self.drop = nn.Dropout(
            dropout
        )

        self.blocks = nn.Sequential(
            *[
                GPTBlock(
                    embed_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.ln_f = nn.LayerNorm(
            embed_dim
        )

        self.head = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False,
        )

        if tie_weights:
            self.head.weight = (
                self.token_embedding.weight
            )

        self.apply(
            self._init_weights
        )

        # Scaled residual initialization.
        #
        # Это соответствует новой финальной NOVA.
        # При загрузке старого checkpoint значения всё равно
        # полностью заменяются весами из checkpoint, поэтому
        # на совместимость со старой моделью это не влияет.
        residual_std = (
            0.02
            / math.sqrt(
                2 * num_layers
            )
        )

        for name, parameter in self.named_parameters():
            if (
                name.endswith(
                    "out_proj.weight"
                )
                or name.endswith(
                    "ffn.2.weight"
                )
            ):
                torch.nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=residual_std,
                )

    def _init_weights(
        self,
        module: nn.Module,
    ):
        if isinstance(
            module,
            nn.Linear,
        ):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                torch.nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def num_params(
        self,
        non_embedding: bool = False,
    ) -> int:
        """
        Число уникальных trainable параметров.

        При tying shared weight считается один раз,
        потому что model.parameters() дедуплицирует
        один и тот же Parameter.
        """

        n = sum(
            p.numel()
            for p in self.parameters()
        )

        if not non_embedding:
            return n

        n -= (
            self.token_embedding
            .weight
            .numel()
        )

        n -= (
            self.position_embedding
            .weight
            .numel()
        )

        if (
            self.head.weight
            is not self.token_embedding.weight
        ):
            n -= (
                self.head
                .weight
                .numel()
            )

        return n

    def weights_are_tied(
        self,
    ) -> bool:
        return (
            self.head.weight
            is self.token_embedding.weight
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        if idx.ndim != 2:
            raise ValueError(
                "idx должен иметь форму "
                "(batch, sequence)"
            )

        batch_size, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} "
                f"> block_size {self.block_size}"
            )

        token_embeddings = (
            self.token_embedding(
                idx
            )
        )

        positions = torch.arange(
            0,
            seq_len,
            device=idx.device,
        ).unsqueeze(0)

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        x = self.drop(
            token_embeddings
            + position_embeddings
        )

        x = self.blocks(
            x
        )

        x = self.ln_f(
            x
        )

        logits = self.head(
            x
        )

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
                ignore_index=self.ignore_index,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Базовая autoregressive generation.

        Более сложная логика assistant:
        - repetition penalty
        - RAG
        - prompt construction
        - stop markers

        будет находиться в inference/assistant.py.
        """

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens должен быть >= 0"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature должен быть > 0"
            )

        self.eval()

        for _ in range(
            max_new_tokens
        ):
            if (
                idx.size(1)
                <= self.block_size
            ):
                idx_cond = idx
            else:
                idx_cond = idx[
                    :,
                    -self.block_size:
                ]

            logits, _ = self(
                idx_cond
            )

            logits = (
                logits[:, -1, :]
                / temperature
            )

            if (
                top_k is not None
                and top_k > 0
            ):
                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                )

                cutoff = values[
                    :,
                    [-1],
                ]

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            if (
                top_p is not None
                and 0.0 < top_p < 1.0
            ):
                sorted_logits, sorted_indices = (
                    torch.sort(
                        logits,
                        descending=True,
                    )
                )

                sorted_probs = F.softmax(
                    sorted_logits,
                    dim=-1,
                )

                cumulative_probs = (
                    torch.cumsum(
                        sorted_probs,
                        dim=-1,
                    )
                )

                remove_mask = (
                    cumulative_probs
                    > top_p
                )

                remove_mask[
                    :,
                    1:
                ] = (
                    remove_mask[
                        :,
                        :-1
                    ]
                    .clone()
                )

                remove_mask[
                    :,
                    0
                ] = False

                sorted_logits = (
                    sorted_logits
                    .masked_fill(
                        remove_mask,
                        float("-inf"),
                    )
                )

                filtered_logits = (
                    torch.full_like(
                        logits,
                        float("-inf"),
                    )
                )

                filtered_logits.scatter_(
                    1,
                    sorted_indices,
                    sorted_logits,
                )

                logits = filtered_logits

            probs = F.softmax(
                logits,
                dim=-1,
            )

            next_token = (
                torch.multinomial(
                    probs,
                    num_samples=1,
                )
            )

            idx = torch.cat(
                [
                    idx,
                    next_token,
                ],
                dim=1,
            )

            if eos_token_id is not None:
                if bool(
                    (
                        next_token
                        == eos_token_id
                    )
                    .all()
                ):
                    break

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (
            0.9,
            0.95,
        ),
        device: str = "cuda",
    ):
        """
        AdamW с двумя группами:

        matrices (dim >= 2)
            weight decay

        vectors/scalars (dim < 2)
            no weight decay
        """

        param_dict = {
            name: parameter
            for name, parameter
            in self.named_parameters()
            if parameter.requires_grad
        }

        decay_params = [
            parameter
            for parameter
            in param_dict.values()
            if parameter.dim() >= 2
        ]

        no_decay_params = [
            parameter
            for parameter
            in param_dict.values()
            if parameter.dim() < 2
        ]

        optim_groups = [
            {
                "params":
                    decay_params,

                "weight_decay":
                    weight_decay,
            },
            {
                "params":
                    no_decay_params,

                "weight_decay":
                    0.0,
            },
        ]

        use_fused = (
            str(device)
            .startswith("cuda")
        )

        try:
            optimizer = (
                torch.optim.AdamW(
                    optim_groups,
                    lr=learning_rate,
                    betas=betas,
                    fused=use_fused,
                )
            )

        except (
            TypeError,
            RuntimeError,
        ):
            optimizer = (
                torch.optim.AdamW(
                    optim_groups,
                    lr=learning_rate,
                    betas=betas,
                )
            )

            use_fused = False

        return optimizer


def extract_model_state_dict(
    checkpoint: Mapping,
) -> Mapping[str, torch.Tensor]:
    """
    Поддерживает два формата:

    1.
        {
            "model_state_dict": {...},
            ...
        }

    2.
        обычный raw state_dict
    """

    if (
        "model_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:
        state_dict = checkpoint

    if not isinstance(
        state_dict,
        Mapping,
    ):
        raise TypeError(
            "Checkpoint не содержит "
            "корректный model state_dict."
        )

    return state_dict


def state_dict_uses_tied_weights(
    state_dict: Mapping[str, torch.Tensor],
) -> bool:
    """
    Определяет, совпадают ли token embedding
    и output head в checkpoint.

    Для новой NOVA эти две матрицы должны
    быть идентичными.

    Для старой NOVA — различными.
    """

    token_weight = state_dict.get(
        "token_embedding.weight"
    )

    head_weight = state_dict.get(
        "head.weight"
    )

    if (
        token_weight is None
        or head_weight is None
    ):
        raise KeyError(
            "В checkpoint отсутствуют "
            "token_embedding.weight "
            "или head.weight."
        )

    if (
        token_weight.shape
        != head_weight.shape
    ):
        return False

    # Если torch.save сохранил shared storage,
    # этот быстрый путь сработает.
    try:
        if (
            token_weight
            .untyped_storage()
            .data_ptr()
            ==
            head_weight
            .untyped_storage()
            .data_ptr()
        ):
            return True
    except Exception:
        pass

    # Универсальная проверка.
    return bool(
        torch.equal(
            token_weight,
            head_weight,
        )
    )


def build_model_for_checkpoint(
    *,
    checkpoint: Mapping,
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    block_size: int,
    dropout: float = 0.0,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> tuple[GPT, bool]:
    """
    Создаёт правильную разновидность NOVA
    для конкретного checkpoint.

    Возвращает:
        model
        tie_weights
    """

    state_dict = (
        extract_model_state_dict(
            checkpoint
        )
    )

    tie_weights = (
        state_dict_uses_tied_weights(
            state_dict
        )
    )

    model = GPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        dropout=dropout,
        tie_weights=tie_weights,
        ignore_index=ignore_index,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return (
        model,
        tie_weights,
    )