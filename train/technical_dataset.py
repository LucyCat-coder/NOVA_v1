from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import Dataset


DEFAULT_EOS_TOKEN_ID = 50256
DEFAULT_IGNORE_INDEX = -100


@dataclass
class EncodedSFTExample:
    input_ids: torch.Tensor
    labels: torch.Tensor

    prompt: str
    answer: str

    prompt_token_count: int
    answer_token_count: int
    supervised_token_count: int

    raw_sequence_ids: tuple[int, ...]

    truncated: bool

    source_line: int


class TechnicalSFTDataset(
    Dataset
):
    """
    Assistant-only SFT dataset для NOVA.

    Формат:

        Пользователь: <question>
        Нова: <answer><EOS>

    ВАЖНО
    ------

    GPT.forward() НЕ делает causal shift.

    Поэтому dataset сам создаёт:

        input_ids = sequence[:-1]
        labels    = sequence[1:]

    После этого labels пользовательской
    части маскируются -100.

    Первый supervised label соответствует
    первому токену ответа.

    Настоящий EOS тоже supervised.

    Padding labels всегда -100.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        block_size: int = 512,
        tokenizer_name: str = "gpt2",
        eos_token_id: int = (
            DEFAULT_EOS_TOKEN_ID
        ),
        ignore_index: int = (
            DEFAULT_IGNORE_INDEX
        ),
    ):
        super().__init__()

        self.path = Path(
            path
        )

        self.block_size = int(
            block_size
        )

        self.eos_token_id = int(
            eos_token_id
        )

        self.ignore_index = int(
            ignore_index
        )

        if self.block_size <= 1:
            raise ValueError(
                "block_size должен быть > 1."
            )

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset не найден: "
                f"{self.path}"
            )

        self.tokenizer = (
            tiktoken.get_encoding(
                tokenizer_name
            )
        )

        if (
            self.eos_token_id
            >= self.tokenizer.n_vocab
        ):
            raise ValueError(
                "EOS token id выходит "
                "за vocab tokenizer."
            )

        self.records = (
            self._read_records()
        )

        if not self.records:
            raise RuntimeError(
                "Dataset пуст."
            )

    # ========================================================
    # JSONL
    # ========================================================

    def _read_records(
        self,
    ) -> list[dict]:
        records: list[dict] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Некорректный JSON "
                        f"в {self.path}, "
                        f"строка {line_number}: "
                        f"{exc}"
                    ) from exc

                question, answer = (
                    self._extract_qa(
                        obj,
                        line_number,
                    )
                )

                records.append(
                    {
                        "question":
                            question,

                        "answer":
                            answer,

                        "source_line":
                            line_number,
                    }
                )

        return records

    @staticmethod
    def _extract_qa(
        obj: dict,
        line_number: int,
    ) -> tuple[str, str]:
        """
        Основной ожидаемый schema:

        {
            "messages": [
                {
                    "role": "user",
                    "content": "..."
                },
                {
                    "role": "assistant",
                    "content": "..."
                }
            ]
        }

        Также поддерживается простой
        fallback question/answer.
        """

        question = None
        answer = None

        messages = obj.get(
            "messages"
        )

        if isinstance(
            messages,
            list,
        ):
            for message in messages:
                if not isinstance(
                    message,
                    dict,
                ):
                    continue

                role = str(
                    message.get(
                        "role",
                        ""
                    )
                ).strip().lower()

                content = message.get(
                    "content"
                )

                if not isinstance(
                    content,
                    str,
                ):
                    continue

                content = (
                    content.strip()
                )

                if (
                    role == "user"
                    and question is None
                ):
                    question = content

                elif (
                    role == "assistant"
                    and answer is None
                ):
                    answer = content

        if question is None:
            value = obj.get(
                "question"
            )

            if isinstance(
                value,
                str,
            ):
                question = (
                    value.strip()
                )

        if answer is None:
            value = obj.get(
                "answer"
            )

            if isinstance(
                value,
                str,
            ):
                answer = (
                    value.strip()
                )

        if not question:
            raise ValueError(
                "Нет user/question "
                f"в строке {line_number}."
            )

        if not answer:
            raise ValueError(
                "Нет assistant/answer "
                f"в строке {line_number}."
            )

        return (
            question,
            answer,
        )

    # ========================================================
    # ENCODING
    # ========================================================

    def encode_example(
        self,
        index: int,
    ) -> EncodedSFTExample:
        record = self.records[
            index
        ]

        question = record[
            "question"
        ]

        answer = record[
            "answer"
        ]

        source_line = int(
            record[
                "source_line"
            ]
        )

        # Важно:
        # inference использует именно
        #
        # Пользователь: <query>
        # Нова:
        #
        # без пробела после двоеточия.
        prompt = (
            f"Пользователь: "
            f"{question}\n"
            f"Нова:"
        )

        prompt_ids = (
            self.tokenizer.encode(
                prompt,
                disallowed_special=(),
            )
        )

        # Первый assistant token должен
        # продолжать "Нова:".
        #
        # Добавляем leading space именно
        # в assistant-side tokenization.
        answer_ids = (
            self.tokenizer.encode(
                " " + answer,
                disallowed_special=(),
            )
        )

        # sequence имеет на один token
        # больше input_ids из-за shift:
        #
        # seq:
        #   prompt + answer + EOS
        #
        # input:
        #   seq[:-1]
        #
        # labels:
        #   seq[1:]
        #
        # Поэтому:
        #
        # len(sequence) <= block_size + 1
        max_answer_tokens = (
            self.block_size
            - len(
                prompt_ids
            )
        )

        if max_answer_tokens <= 0:
            raise ValueError(
                "Prompt слишком длинный "
                "для block_size="
                f"{self.block_size}. "
                f"Dataset={self.path}, "
                f"line={source_line}, "
                f"prompt_tokens="
                f"{len(prompt_ids)}"
            )

        truncated = (
            len(answer_ids)
            > max_answer_tokens
        )

        if truncated:
            answer_ids = (
                answer_ids[
                    :max_answer_tokens
                ]
            )

        if not answer_ids:
            raise ValueError(
                "После tokenization "
                "assistant answer пуст. "
                f"line={source_line}"
            )

        sequence_ids = (
            prompt_ids
            + answer_ids
            + [
                self.eos_token_id
            ]
        )

        if (
            len(sequence_ids)
            > self.block_size + 1
        ):
            raise AssertionError(
                "Internal length error."
            )

        input_ids = (
            sequence_ids[:-1]
        )

        labels = (
            sequence_ids[1:]
        )

        # ----------------------------------------------------
        # ASSISTANT-ONLY MASK
        # ----------------------------------------------------
        #
        # labels[i] = sequence[i + 1]
        #
        # Первый токен assistant answer
        # находится:
        #
        # sequence[len(prompt_ids)]
        #
        # Значит он предсказывается
        # позицией:
        #
        # labels[len(prompt_ids) - 1]
        #
        # Всё ДО этой позиции должно
        # быть ignore_index.
        first_supervised_position = (
            len(prompt_ids) - 1
        )

        labels = list(
            labels
        )

        for position in range(
            first_supervised_position
        ):
            labels[
                position
            ] = (
                self.ignore_index
            )

        supervised_before_padding = sum(
            token != self.ignore_index
            for token in labels
        )

        expected_supervised = (
            len(answer_ids)
            + 1
        )

        if (
            supervised_before_padding
            != expected_supervised
        ):
            raise AssertionError(
                "Ошибка assistant-only mask: "
                f"{supervised_before_padding} "
                f"!= {expected_supervised}"
            )

        # Последний supervised target
        # обязан быть НАСТОЯЩИМ EOS.
        if (
            labels[-1]
            != self.eos_token_id
        ):
            raise AssertionError(
                "EOS не supervised."
            )

        # ----------------------------------------------------
        # FIXED-LENGTH PADDING
        # ----------------------------------------------------

        padding = (
            self.block_size
            - len(input_ids)
        )

        if padding < 0:
            raise AssertionError(
                "input_ids длиннее block_size."
            )

        if padding:
            # GPT-2 не имеет отдельного PAD.
            #
            # Для input_ids используем EOS,
            # но padding targets всегда -100.
            input_ids = (
                input_ids
                + [
                    self.eos_token_id
                ]
                * padding
            )

            labels = (
                labels
                + [
                    self.ignore_index
                ]
                * padding
            )

        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
        )

        label_tensor = torch.tensor(
            labels,
            dtype=torch.long,
        )

        if (
            input_tensor.shape
            != (
                self.block_size,
            )
        ):
            raise AssertionError(
                "Некорректная форма "
                "input_ids."
            )

        if (
            label_tensor.shape
            != (
                self.block_size,
            )
        ):
            raise AssertionError(
                "Некорректная форма "
                "labels."
            )

        supervised_token_count = int(
            (
                label_tensor
                != self.ignore_index
            )
            .sum()
            .item()
        )

        return EncodedSFTExample(
            input_ids=input_tensor,
            labels=label_tensor,
            prompt=prompt,
            answer=answer,
            prompt_token_count=(
                len(prompt_ids)
            ),
            answer_token_count=(
                len(answer_ids)
            ),
            supervised_token_count=(
                supervised_token_count
            ),
            raw_sequence_ids=tuple(
                sequence_ids
            ),
            truncated=truncated,
            source_line=source_line,
        )

    # ========================================================
    # PYTORCH DATASET
    # ========================================================

    def __len__(
        self,
    ) -> int:
        return len(
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        example = (
            self.encode_example(
                index
            )
        )

        return {
            "input_ids":
                example.input_ids,

            "labels":
                example.labels,
        }