from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from model import build_model_for_checkpoint


class Assistant:
    EOS_TOKEN_ID = 50256

    VALID_RAG_SOURCES = {
        "auto",
        "core",
        "technical",
        "broad",
        "off",
    }

    VALID_ANSWER_MODES = {
        "auto",
        "direct",
        "generate",
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_config: dict,
        device: str = "cuda",
        rag_core_index_path: str | Path | None = None,
        rag_technical_index_path: str | Path | None = None,
        rag_broad_index_path: str | Path | None = None,
        rag_core_threshold: float = 0.83,
        rag_index_path: str | Path | None = None,
    ):
        """
        NOVA inference wrapper.

        RAG sources
        -----------

        auto:
            SAFE production route.

            Ищет только в CORE.

            Если CORE уверен:
                возвращается проверенный
                retrieved answer напрямую.

            Если CORE не уверен:
                FINAL NOVA генерирует сама
                без RAG context.

        core:
            Ручной поиск в CORE.

        technical:
            Ручной поиск в technical_v1_1.

        broad:
            Ручной экспериментальный поиск
            в knowledge_index_v2.

        off:
            RAG отключён.

        Answer modes
        ------------

        auto:
            CORE / TECHNICAL
                -> direct retrieved answer

            BROAD
                -> retrieved context
                   передается FINAL NOVA

            NONE / OFF
                -> обычная генерация NOVA

        direct:
            Если retrieval что-то нашёл,
            первый retrieved text возвращается
            напрямую независимо от source.

        generate:
            Всегда запускает FINAL NOVA.
            Retrieved text, если есть,
            используется как prompt context.

        Backward compatibility
        ----------------------

        rag_index_path сохранён для старого
        app.py.

        Если rag_broad_index_path не задан,
        rag_index_path трактуется как broad.
        """

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.model_config = dict(
            model_config
        )

        self.device = torch.device(
            device
        )

        self.tokenizer = (
            tiktoken.get_encoding(
                "gpt2"
            )
        )

        if (
            self.tokenizer.n_vocab
            != self.model_config[
                "vocab_size"
            ]
        ):
            raise RuntimeError(
                "Tokenizer vocab не совпадает "
                "с конфигом: "
                f"{self.tokenizer.n_vocab} != "
                f"{self.model_config['vocab_size']}"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint не найден: "
                f"{self.checkpoint_path}"
            )

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "В конфиге указана CUDA, "
                "но torch.cuda.is_available() "
                "== False"
            )

        # ====================================================
        # RAG CONFIGURATION
        # ====================================================

        self.rag_core_index_path = (
            Path(
                rag_core_index_path
            )
            if rag_core_index_path
            is not None
            else None
        )

        self.rag_technical_index_path = (
            Path(
                rag_technical_index_path
            )
            if rag_technical_index_path
            is not None
            else None
        )

        broad_path = (
            rag_broad_index_path
            if rag_broad_index_path
            is not None
            else rag_index_path
        )

        self.rag_broad_index_path = (
            Path(
                broad_path
            )
            if broad_path is not None
            else None
        )

        # Старое имя оставляем,
        # чтобы старый app.py пока
        # не ломался.
        self.rag_index_path = (
            self.rag_broad_index_path
        )

        self.rag_core_threshold = float(
            rag_core_threshold
        )

        if not (
            0.0
            <= self.rag_core_threshold
            <= 1.0
        ):
            raise ValueError(
                "rag_core_threshold должен "
                "быть между 0 и 1."
            )

        self._safe_retriever = None

        self._warned_reranker = False

        # ====================================================
        # MODEL
        # ====================================================

        print("=" * 72)
        print("NOVA Assistant")
        print("=" * 72)

        print(
            f"Checkpoint: "
            f"{self.checkpoint_path}"
        )

        print(
            f"Device:     "
            f"{self.device}"
        )

        checkpoint = (
            self._load_checkpoint(
                self.checkpoint_path
            )
        )

        (
            self.model,
            self.tie_weights,
        ) = build_model_for_checkpoint(
            checkpoint=checkpoint,
            vocab_size=(
                self.model_config[
                    "vocab_size"
                ]
            ),
            embed_dim=(
                self.model_config[
                    "embed_dim"
                ]
            ),
            num_heads=(
                self.model_config[
                    "num_heads"
                ]
            ),
            num_layers=(
                self.model_config[
                    "num_layers"
                ]
            ),
            block_size=(
                self.model_config[
                    "block_size"
                ]
            ),
            dropout=0.0,
        )

        self.iter_num = (
            checkpoint.get(
                "iter_num",
                None,
            )
            if isinstance(
                checkpoint,
                dict,
            )
            else None
        )

        self.best_val_loss = (
            checkpoint.get(
                "best_val_loss",
                None,
            )
            if isinstance(
                checkpoint,
                dict,
            )
            else None
        )

        del checkpoint

        gc.collect()

        self.model.to(
            self.device
        )

        self.model.eval()

        self.block_size = (
            self.model.block_size
        )

        print(
            "Architecture: "
            + (
                "TIED"
                if self.tie_weights
                else "UNTIED"
            )
        )

        print(
            "Parameters:   "
            f"{self.model.num_params() / 1_000_000:.3f}M"
        )

        if self.iter_num is not None:
            print(
                f"iter_num:     "
                f"{self.iter_num}"
            )

        if self.best_val_loss is not None:
            print(
                "best_val_loss:"
                f"{self.best_val_loss:.6f}"
            )

        if self.device.type == "cuda":
            print(
                "GPU:          "
                f"{torch.cuda.get_device_name(self.device)}"
            )

            print(
                "BF16:         "
                f"{torch.cuda.is_bf16_supported()}"
            )

        self._warmup()

        print(
            "CORE gate:    "
            f"{self.rag_core_threshold:.3f}"
        )

        print(
            "AUTO:         "
            "CORE direct -> fallback model"
        )

        print(
            "TECHNICAL:    "
            "manual direct"
        )

        print(
            "BROAD:        "
            "manual experimental"
        )

        print(
            "Assistant ready."
        )

        print("=" * 72)

    # ========================================================
    # CHECKPOINT
    # ========================================================

    @staticmethod
    def _load_checkpoint(
        path: Path,
    ):
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

        except Exception as exc:
            print(
                "[WARN] weights_only=True "
                "не сработал:"
            )

            print(
                "       "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "[WARN] Повторная загрузка "
                "weights_only=False."
            )

            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

    # ========================================================
    # DEVICE / AUTOCAST
    # ========================================================

    def _autocast_context(
        self,
    ):
        if self.device.type != "cuda":
            return nullcontext()

        if torch.cuda.is_bf16_supported():
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )

        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    @torch.inference_mode()
    def _warmup(
        self,
    ):
        dummy = torch.zeros(
            (1, 1),
            dtype=torch.long,
            device=self.device,
        )

        with self._autocast_context():
            logits, _ = self.model(
                dummy
            )

        if not torch.isfinite(
            logits
        ).all():
            raise RuntimeError(
                "Warmup модели вернул "
                "NaN/Inf."
            )

    # ========================================================
    # SAFE RAG
    # ========================================================

    @property
    def safe_retriever(
        self,
    ):
        """
        SafeRetriever загружается лениво.

        Пока RAG не используется,
        embedding model и FAISS indexes
        не загружаются.
        """

        if (
            self._safe_retriever
            is not None
        ):
            return self._safe_retriever

        try:
            from rag import SafeRetriever

        except (
            ImportError,
            AttributeError,
        ) as exc:
            raise RuntimeError(
                "SafeRetriever не найден. "
                "Проверь rag/safe_retriever.py "
                "и rag/__init__.py."
            ) from exc

        kwargs = {
            "device": str(
                self.device
            ),
            "core_threshold": (
                self.rag_core_threshold
            ),
        }

        if (
            self.rag_core_index_path
            is not None
        ):
            kwargs[
                "core_index_path"
            ] = (
                self.rag_core_index_path
            )

        if (
            self.rag_technical_index_path
            is not None
        ):
            kwargs[
                "technical_index_path"
            ] = (
                self.rag_technical_index_path
            )

        if (
            self.rag_broad_index_path
            is not None
        ):
            kwargs[
                "broad_index_path"
            ] = (
                self.rag_broad_index_path
            )

        print(
            "Loading SafeRetriever..."
        )

        self._safe_retriever = (
            SafeRetriever(
                **kwargs
            )
        )

        print(
            "SafeRetriever ready."
        )

        return self._safe_retriever

    @property
    def retriever(
        self,
    ):
        """
        Backward-compatible alias.
        """

        return self.safe_retriever

    # ========================================================
    # PROMPTS
    # ========================================================

    def _plain_prompt(
        self,
        query: str,
    ) -> str:
        # После "Нова:"
        # специально нет пробела.
        return (
            f"Пользователь: {query}\n"
            f"Нова:"
        )

    def _build_rag_prompt(
        self,
        query: str,
        contexts: list[str],
        mode: str,
        max_prompt_tokens: int,
    ) -> str:
        """
        Формирует старый RAG prompt.

        Сейчас этот путь используется
        в основном для ручного BROAD
        и диагностического
        answer_mode="generate".

        CORE и TECHNICAL в production
        возвращаются напрямую.
        """

        if not contexts:
            return self._plain_prompt(
                query
            )

        context_text = "\n\n".join(
            contexts
        )

        if mode == "inline":
            prefix = (
                f"Пользователь: {query}\n\n"
                f"Известно, что: "
            )

            suffix = (
                "\n\nНова:"
            )

        elif mode == "naive":
            prefix = (
                f"Пользователь: {query}\n"
                f"Контекст:\n"
            )

            suffix = (
                "\n\nНова:"
            )

        elif mode == "prefix":
            prefix = ""

            suffix = (
                "\n\n"
                f"Пользователь: {query}\n"
                f"Нова:"
            )

        else:
            raise ValueError(
                "Неизвестный rag_mode: "
                f"{mode!r}. "
                "Допустимо: "
                "inline, naive, prefix."
            )

        prefix_ids = (
            self.tokenizer.encode(
                prefix,
                disallowed_special=(),
            )
        )

        suffix_ids = (
            self.tokenizer.encode(
                suffix,
                disallowed_special=(),
            )
        )

        available_context_tokens = (
            max_prompt_tokens
            - len(prefix_ids)
            - len(suffix_ids)
        )

        if (
            available_context_tokens
            <= 0
        ):
            return self._plain_prompt(
                query
            )

        context_ids = (
            self.tokenizer.encode(
                context_text,
                disallowed_special=(),
            )
        )

        if (
            len(context_ids)
            > available_context_tokens
        ):
            context_ids = (
                context_ids[
                    :available_context_tokens
                ]
            )

        trimmed_context = (
            self.tokenizer.decode(
                context_ids
            )
        )

        return (
            prefix
            + trimmed_context
            + suffix
        )

    # ========================================================
    # SAMPLING
    # ========================================================

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_token_ids: list[int],
        repetition_penalty: float,
    ) -> torch.Tensor:
        """
        Штрафуем только уже сгенерированные
        NOVA токены.

        Токены вопроса и prompt context
        не штрафуются.
        """

        if (
            repetition_penalty == 1.0
            or not generated_token_ids
        ):
            return logits

        if repetition_penalty <= 0:
            raise ValueError(
                "repetition_penalty "
                "должен быть > 0"
            )

        for token_id in set(
            generated_token_ids
        ):
            value = logits[
                0,
                token_id,
            ]

            if value > 0:
                logits[
                    0,
                    token_id,
                ] = (
                    value
                    / repetition_penalty
                )

            else:
                logits[
                    0,
                    token_id,
                ] = (
                    value
                    * repetition_penalty
                )

        return logits

    @staticmethod
    def _apply_top_k(
        logits: torch.Tensor,
        top_k: int | None,
    ) -> torch.Tensor:
        if (
            top_k is None
            or top_k <= 0
        ):
            return logits

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

        return logits.masked_fill(
            logits < cutoff,
            float("-inf"),
        )

    @staticmethod
    def _apply_top_p(
        logits: torch.Tensor,
        top_p: float | None,
    ) -> torch.Tensor:
        if (
            top_p is None
            or top_p <= 0
            or top_p >= 1
        ):
            return logits

        (
            sorted_logits,
            sorted_indices,
        ) = torch.sort(
            logits,
            descending=True,
            dim=-1,
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
            1:,
        ] = (
            remove_mask[
                :,
                :-1,
            ].clone()
        )

        remove_mask[
            :,
            0,
        ] = False

        sorted_logits = (
            sorted_logits.masked_fill(
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

        return filtered_logits

    # ========================================================
    # GENERATION
    # ========================================================

    @torch.inference_mode()
    def _generate_tokens(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
        seed: int | None,
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []

        if temperature < 0:
            raise ValueError(
                "temperature должен "
                "быть >= 0"
            )

        generated = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        new_tokens: list[int] = []

        generator = None

        if seed is not None:
            generator_device = (
                "cuda"
                if self.device.type
                == "cuda"
                else "cpu"
            )

            generator = (
                torch.Generator(
                    device=generator_device
                )
            )

            generator.manual_seed(
                seed
            )

        for _ in range(
            max_new_tokens
        ):
            idx_cond = generated[
                :,
                -self.block_size:,
            ]

            with self._autocast_context():
                logits, _ = self.model(
                    idx_cond
                )

            next_logits = (
                logits[
                    :,
                    -1,
                    :,
                ]
                .float()
            )

            if not torch.isfinite(
                next_logits
            ).all():
                raise RuntimeError(
                    "Модель вернула "
                    "NaN/Inf logits."
                )

            next_logits = (
                self._apply_repetition_penalty(
                    next_logits,
                    new_tokens,
                    repetition_penalty,
                )
            )

            if temperature == 0:
                next_token = (
                    torch.argmax(
                        next_logits,
                        dim=-1,
                        keepdim=True,
                    )
                )

            else:
                next_logits = (
                    next_logits
                    / temperature
                )

                next_logits = (
                    self._apply_top_k(
                        next_logits,
                        top_k,
                    )
                )

                next_logits = (
                    self._apply_top_p(
                        next_logits,
                        top_p,
                    )
                )

                probs = F.softmax(
                    next_logits,
                    dim=-1,
                )

                next_token = (
                    torch.multinomial(
                        probs,
                        num_samples=1,
                        generator=generator,
                    )
                )

            token_id = int(
                next_token.item()
            )

            if (
                token_id
                == self.EOS_TOKEN_ID
            ):
                break

            new_tokens.append(
                token_id
            )

            generated = torch.cat(
                [
                    generated,
                    next_token,
                ],
                dim=1,
            )

        return new_tokens

    def _decode_generated(
        self,
        token_ids: list[int],
    ) -> str:
        """
        GPT-2 tokenizer byte-level.

        decode_bytes + errors='ignore'
        не оставляет символ �,
        если generation закончилась
        посреди UTF-8 последовательности.
        """

        if not token_ids:
            return ""

        raw = (
            self.tokenizer.decode_bytes(
                token_ids
            )
        )

        text = raw.decode(
            "utf-8",
            errors="ignore",
        )

        return text

    @staticmethod
    def _clean_answer(
        answer: str,
    ) -> str:
        answer = answer.strip()

        stop_markers = [
            "\nПользователь:",
            "\n\nПользователь:",
            "\nКонтекст:",
            "\nИзвестно, что:",
        ]

        positions = []

        for marker in stop_markers:
            pos = answer.find(
                marker
            )

            if pos >= 0:
                positions.append(
                    pos
                )

        if positions:
            answer = answer[
                :min(
                    positions
                )
            ].strip()

        return answer

    # ========================================================
    # ANSWER ROUTING
    # ========================================================

    @staticmethod
    def _should_use_direct_answer(
        *,
        answer_mode: str,
        rag_route: str,
        contexts: list[str],
    ) -> bool:
        if not contexts:
            return False

        if answer_mode == "direct":
            return True

        if answer_mode == "generate":
            return False

        # answer_mode == "auto"
        #
        # CORE и TECHNICAL считаем
        # доверенными слоями.
        #
        # BROAD не считаем доверенным
        # автоматически.
        return rag_route in {
            "core",
            "technical",
        }

    # ========================================================
    # PUBLIC GENERATE
    # ========================================================

    def generate(
        self,
        query: str,
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.5,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
        repetition_penalty: float = 1.15,
        seed: int | None = None,
        rag_enabled: bool = False,
        rag_source: str = "auto",
        rag_mode: str = "inline",
        answer_mode: str = "auto",
        top_k_docs: int = 1,
        min_score: float | None = 0.80,
        use_reranker: bool = False,
        return_context: bool = False,
        return_rag_info: bool = False,
    ):
        """
        Основной inference entry point.

        answer_mode="auto"
        ------------------

        AUTO + CORE accepted:
            direct CORE answer.

        TECHNICAL manual:
            direct technical answer.

        BROAD manual:
            model generation with broad context.

        AUTO rejected:
            plain FINAL NOVA generation.

        answer_mode="direct"
        --------------------

        Любой найденный retrieval result
        возвращается напрямую.

        answer_mode="generate"
        ----------------------

        Всегда используется генератор.
        Если retrieval context найден,
        он добавляется в prompt.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Пустой запрос."
            )

        rag_source = str(
            rag_source
        ).strip().lower()

        if (
            rag_source
            not in self.VALID_RAG_SOURCES
        ):
            raise ValueError(
                "Неизвестный rag_source: "
                f"{rag_source!r}. "
                "Допустимо: "
                + ", ".join(
                    sorted(
                        self.VALID_RAG_SOURCES
                    )
                )
            )

        answer_mode = str(
            answer_mode
        ).strip().lower()

        if (
            answer_mode
            not in self.VALID_ANSWER_MODES
        ):
            raise ValueError(
                "Неизвестный answer_mode: "
                f"{answer_mode!r}. "
                "Допустимо: "
                + ", ".join(
                    sorted(
                        self.VALID_ANSWER_MODES
                    )
                )
            )

        if (
            max_new_tokens
            >= self.block_size
        ):
            raise ValueError(
                "max_new_tokens должен "
                "быть меньше block_size."
            )

        if top_k_docs <= 0:
            raise ValueError(
                "top_k_docs должен быть > 0."
            )

        contexts: list[str] = []

        rag_info = {
            "mode": "off",
            "route": "off",
            "accepted": False,
            "top1_score": None,
            "threshold": None,
            "results": [],
        }

        if (
            use_reranker
            and not self._warned_reranker
        ):
            print(
                "[WARN] use_reranker=True "
                "игнорируется: SafeRetriever "
                "работает без reranker."
            )

            self._warned_reranker = True

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        if (
            rag_enabled
            and rag_source != "off"
        ):
            rag_info = (
                self.safe_retriever
                .retrieve_with_info(
                    query=query,
                    mode=rag_source,
                    top_k=top_k_docs,
                    min_score=min_score,
                )
            )

            # Работаем с копией:
            # ниже добавим поля,
            # относящиеся к Assistant.
            rag_info = dict(
                rag_info
            )

            contexts = [
                text
                for (
                    text,
                    _score,
                    _metadata,
                )
                in rag_info[
                    "results"
                ]
            ]

        elif rag_source == "off":
            rag_info = {
                "mode": "off",
                "route": "off",
                "accepted": False,
                "top1_score": None,
                "threshold": None,
                "results": [],
            }

        rag_route = str(
            rag_info.get(
                "route",
                "none",
            )
        )

        # ----------------------------------------------------
        # DIRECT GROUNDED ANSWER
        # ----------------------------------------------------

        use_direct = (
            rag_enabled
            and self._should_use_direct_answer(
                answer_mode=answer_mode,
                rag_route=rag_route,
                contexts=contexts,
            )
        )

        if use_direct:
            answer = (
                contexts[0]
                .strip()
            )

            rag_info[
                "answer_mode"
            ] = answer_mode

            rag_info[
                "answer_source"
            ] = "retrieval"

            rag_info[
                "direct_answer"
            ] = True

            rag_info[
                "model_generated"
            ] = False

            if return_rag_info:
                return (
                    answer,
                    contexts,
                    rag_info,
                )

            if return_context:
                return (
                    answer,
                    contexts,
                )

            return answer

        # ----------------------------------------------------
        # MODEL GENERATION
        # ----------------------------------------------------

        max_prompt_tokens = (
            self.block_size
            - max_new_tokens
        )

        if max_prompt_tokens <= 0:
            raise ValueError(
                "Для prompt "
                "не осталось места."
            )

        if (
            rag_enabled
            and contexts
        ):
            prompt = (
                self._build_rag_prompt(
                    query=query,
                    contexts=contexts,
                    mode=rag_mode,
                    max_prompt_tokens=(
                        max_prompt_tokens
                    ),
                )
            )

        else:
            prompt = (
                self._plain_prompt(
                    query
                )
            )

        input_ids = (
            self.tokenizer.encode(
                prompt,
                disallowed_special=(),
            )
        )

        if (
            len(input_ids)
            > max_prompt_tokens
        ):
            # Сохраняем конец prompt,
            # где находится "Нова:".
            input_ids = input_ids[
                -max_prompt_tokens:
            ]

        new_ids = (
            self._generate_tokens(
                input_ids=input_ids,
                max_new_tokens=(
                    max_new_tokens
                ),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=(
                    repetition_penalty
                ),
                seed=seed,
            )
        )

        answer = (
            self._decode_generated(
                new_ids
            )
        )

        answer = self._clean_answer(
            answer
        )

        rag_info = dict(
            rag_info
        )

        rag_info[
            "answer_mode"
        ] = answer_mode

        rag_info[
            "answer_source"
        ] = "model"

        rag_info[
            "direct_answer"
        ] = False

        rag_info[
            "model_generated"
        ] = True

        if return_rag_info:
            return (
                answer,
                contexts,
                rag_info,
            )

        if return_context:
            return (
                answer,
                contexts,
            )

        return answer