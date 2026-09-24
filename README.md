# NOVA_v1

> ****Статус:**** архивная экспериментальная версия / baseline  

> ****Дата заморозки:**** 24 сентября 2026  

> ****Следующая версия:**** `NOVA_v2`  

> ****Основной язык документации:**** русский  

> ****Тип проекта:**** собственная decoder-only LLM + локальный AI-ассистент + RAG-эксперименты

---

## 1. Что такое NOVA_v1

****NOVA_v1**** — это первая законченная экспериментальная версия моей собственной локальной языковой модели NOVA.

Изначальная цель проекта — не просто взять готовую большую модель вроде Qwen/Llama/Mistral, немного дообучить её и назвать своей. Идея NOVA — постепенно создать ****собственную LLM с нуля**** и построить вокруг неё персонального помощника со своим характером, технической специализацией, памятью, RAG, инструментами и в будущем голосовым интерфейсом.

NOVA_v1 стала первым полным инженерным циклом этой идеи:

```text

данные

  ↓

токенизация

  ↓

собственная GPT-подобная модель

  ↓

pretraining

  ↓

checkpoint

  ↓

inference

  ↓

RAG

  ↓

SafeRetriever

  ↓

SFT

  ↓

evaluation

  ↓

диагностика ограничений

```

Важно: ****NOVA_v1 не является готовым надёжным ассистентом****.

Она сохранена как исследовательская версия и baseline, потому что именно на ней были обнаружены важные ограничения первого подхода и сформулированы требования к `NOVA_v2`.

Подробный технический итог проекта находится в:

```text

NOVA_v1_FINAL_REPORT.md

```

Если нужно быстро понять проект с нуля, рекомендуется читать файлы в таком порядке:

```text

1\. README.md

2\. NOVA_v1_FINAL_REPORT.md

3\. reports/sft_teacher_forcing_diagnostic.txt

4\. reports/final_quality_audit.txt

5\. reports/decoding_comparison.txt

```

---

# 2. Главный результат NOVA_v1

NOVA_v1 технически научилась:

\- загружать собственный checkpoint;

\- генерировать текст локально;

\- работать на CUDA;

\- использовать BF16;

\- выполнять pretraining;

\- выполнять assistant-only SFT;

\- использовать FAISS;

\- использовать multilingual E5 embeddings;

\- работать с несколькими RAG-индексами;

\- маршрутизировать запросы через SafeRetriever;

\- возвращать trusted RAG-ответ напрямую;

\- работать через локальный Gradio-интерфейс;

\- проходить набор автоматических и ручных тестов.

Но независимые тесты показали, что модель пока ****ненадёжна как самостоятельный универсальный ассистент****.

Главный экспериментальный вывод:

```text

SFT уменьшает teacher-forced loss

и улучшает token accuracy,

но свободная autoregressive generation

остаётся нестабильной.

```

Поэтому дальнейшее «латание» v1 остановлено.

Следующая версия — `NOVA_v2` — будет проектироваться заново с учётом результатов v1.

---

# 3. Долгосрочная цель NOVA

Планируемая система должна состоять из нескольких независимых уровней.

## 3.1 NOVA Core

Собственная базовая языковая модель:

\- decoder-only Transformer;

\- русский язык как основной;

\- английский при необходимости;

\- генерация текста;

\- базовые знания;

\- понимание общего языка;

\- техническая база.

## 3.2 NOVA Assistant

Поведение ассистента:

\- следование инструкциям;

\- помощь в обучении;

\- помощь с Python и AI-разработкой;

\- объяснение сложного простыми словами;

\- умение признавать неопределённость;

\- нормальный диалог.

## 3.3 NOVA Personality

Стабильный стиль и характер:

\- спокойствие;

\- рассудительность;

\- любознательность;

\- честность;

\- уважительный тон;

\- отсутствие снисходительности;

\- мягкий юмор;

\- способность обсуждать и технические, и обычные темы.

## 3.4 Memory

Изменяемая память должна быть отдельной системой, а не постоянным переобучением весов.

Например:

\- факты о пользователе;

\- история проектов;

\- заметки;

\- предпочтения;

\- важные события;

\- контекст прошлых разговоров.

## 3.5 RAG

Внешние знания:

\- техническая документация;

\- доверенные факты;

\- пользовательские документы;

\- базы знаний;

\- заметки.

## 3.6 Tools / Agent

В будущем:

\- чтение файлов;

\- поиск по проекту;

\- запуск Python;

\- работа с кодом;

\- безопасное изменение файлов;

\- другие локальные инструменты.

## 3.7 Voice

Планируемая схема:

```text

speech-to-text

      ↓

NOVA

      ↓

text-to-speech

```

---

# 4. Что представляет собой модель NOVA_v1

Финальная базовая модель — собственный GPT-подобный decoder-only Transformer.

Основные параметры:

```text

vocab_size      = 50 257

embedding size  = 768

attention heads = 12

layers          = 12

context         = 512 tokens

weight tying    = yes

parameters      = 124 009 728

```

То есть приблизительно:

```text

124M параметров

```

Это не Hugging Face causal-LM checkpoint и не LoRA поверх готовой модели.

Архитектура реализована внутри проекта на PyTorch.

---

# 5. Архитектура модели

Основные файлы:

```text

model/

├── \_\_init\_\_.py

├── attention.py

└── model.py

```

## `model/attention.py`

Содержит causal self-attention.

Используется PyTorch scaled dot-product attention.

## `model/model.py`

Содержит:

\- `GPTBlock`;

\- `GPT`;

\- token embeddings;

\- learned position embeddings;

\- LayerNorm;

\- feed-forward blocks;

\- GELU;

\- output head;

\- weight tying;

\- autoregressive generation;

\- конфигурацию AdamW;

\- загрузку checkpoint;

\- проверку tied/untied weights.

Архитектура использует pre-norm блоки:

```text

x

│

├── LayerNorm

├── causal attention

└── residual

       ↓

├── LayerNorm

├── FFN

└── residual

```

Feed-forward часть:

```text

768

 ↓

3072

 ↓ GELU

768

```

---

# 6. Tokenizer

NOVA_v1 использует:

```python

tiktoken.get_encoding("gpt2")

```

Это GPT-2 byte-level BPE.

Размер словаря:

```text

50 257 tokens

```

Плюсы:

\- стабильная готовая реализация;

\- можно кодировать произвольный Unicode;

\- нет классической проблемы неизвестного `\<UNK>`;

\- хорошо подходит по формату к GPT-подобной архитектуре.

Минус для нашей задачи:

GPT-2 tokenizer не создавался специально под русский язык.

Русские слова могут разбиваться на менее эффективные byte/subword последовательности.

Важно: это ****не доказано как единственная причина проблем NOVA_v1****. Это одна из гипотез, которую планируется отдельно проверить в `NOVA_v2`.

Для NOVA_v2 предполагается исследовать tokenizer на смеси:

```text

русский

\+

английский

\+

Python/code

\+

технические тексты

```

---

# 7. Context length

Контекст NOVA_v1:

```text

512 tokens

```

Для короткого вопроса этого хватает.

Но для будущего ассистента 512 токенов мало, особенно когда в prompt одновременно входят:

\- вопрос пользователя;

\- кусок кода;

\- traceback;

\- RAG-контекст;

\- история диалога;

\- системные инструкции.

Поэтому увеличение context length является одним из направлений NOVA_v2.

---

# 8. Данные для pretraining

Фактическое состояние архива NOVA_v1 на момент подготовки к публикации:

```text
data/
├── pretrain_final/
│   └── train.txt
└── val.txt
```

Размеры файлов:

```text
data/pretrain_final/train.txt: 116 602 874 bytes
data/val.txt:                   18 008 668 bytes
```

SHA-256 текущего validation corpus:

```text
25BED3A3C6543EBBB94B42AD9297FC9095A48C89FC7CAD61B4E9EDDE2F18F692
```

Train corpus содержит приблизительно:

```text
66.6M GPT-2 tokens
```

Важно: в финальном архиве **нет** файла:

```text
data/pretrain_final/val.txt
```

Validation corpus физически находится в:

```text
data/val.txt
```

Дополнительная проверка показала, что два файла, которые могли бы хранить историческую конфигурацию подготовки pretraining data,

```text
config/training/pretrain_config.py
scripts/prepare_pretrain_data.py
```

на момент архивации существуют, но имеют размер:

```text
0 bytes
```

Поэтому по сохранённому состоянию NOVA_v1 невозможно достоверно восстановить, каким именно способом путь к validation corpus передавался во время первоначального pretraining run.

Документация намеренно фиксирует **то, что реально сохранилось**, и не заполняет недостающую историческую информацию предположениями.

В ранних версиях проекта использовалась смесь:

- knowledge data;
- synthetic data;
- Alpaca/instruction-style data;
- автоматически сгенерированных диалогов;
- технических текстов.

Один из главных выводов NOVA_v1:

```text
PRETRAINING DATA
        ≠
ASSISTANT SFT DATA
        ≠
PERSONALITY DATA
        ≠
RAG KNOWLEDGE
```

В NOVA_v2 эти типы данных должны быть разделены намного строже.

Большие pretraining-файлы не публикуются в GitHub и исключены через `.gitignore`.

---

# 9. Почему большие данные и веса не публикуются вместе с кодом

GitHub-репозиторий нужен прежде всего для:

\- исходного кода;

\- конфигурации;

\- тестов;

\- скриптов;

\- документации;

\- небольших воспроизводимых datasets;

\- текстовых reports.

В GitHub ****не должны попадать****:

\- `.venv`;

\- большие `.pt` checkpoints;

\- FAISS binary indices;

\- большие pretraining corpora;

\- caches;

\- временные файлы;

\- секреты/API keys;

\- персональная память;

\- приватные документы.

Поэтому часть файлов, описанных в README, будет отсутствовать после обычного `git clone`.

Это сделано намеренно.

---

# 10. Checkpoints

## 10.1 FINAL checkpoint

Локальный путь внутри проекта:

```text

out/final/best_final.pt

```

SHA-256:

```text

765E3B02A8DA32E30D728BBE6A68F7B5130A5FA6726B0BA5775F22352D83231E

```

Метаданные:

```text

iter_num      = 45000

best_val_loss = 0.515568

architecture  = tied weights

parameters    ≈ 124.010M

```

## 10.2 SFT — 20 steps

```text

out/candidate_sft/technical_v1_step20/best_candidate.pt

```

SHA-256:

```text

19B88F804126B085AE70830E2654589EED2899E76DE65804968CE331302EDE90

```

Это экспериментальный checkpoint.

Он не заменяет FINAL.

## 10.3 SFT — 120 steps

```text

out/candidate_sft/technical_v1_step120/best_candidate.pt

```

SHA-256:

```text

3C1C3D89C81B16EDF6C982210BEDB33B04512FE25B7DF9BBFF6BFC10DDDD36A8

```

Метаданные:

```text

iter_num ≈ 45120

SFT validation loss ≈ 1.0968

```

Это тоже экспериментальный checkpoint.

---

# 11. Почему checkpoints не хранятся в Git

Checkpoints:

\- большие;

\- бинарные;

\- плохо подходят для обычной Git-истории;

\- создают огромные binary diff;

\- могут превысить ограничения GitHub.

Поэтому `.pt`, `.pth`, `.ckpt`, `.safetensors` исключаются через `.gitignore`.

Если понадобится отдельно опубликовать веса, лучше использовать:

\- GitHub Release;

\- Hugging Face Hub;

\- облачное хранилище;

\- model registry.

---

# 12. Pretraining

Основная финальная конфигурация:

```text

batch_size                  = 4

gradient_accumulation_steps = 8

effective batch             = 32 sequences

learning_rate = 1e-4

min_lr        = 1e-5

weight_decay = 0.1

betas        = (0.9, 0.95)

grad_clip    = 1.0

max_iters     = 50000

eval_interval = 1000

eval_iters    = 50

save_interval = 2500

dtype  = bfloat16

device = cuda

```

Лучший сохранённый checkpoint:

```text

iteration 45000

best validation loss ≈ 0.515568

```

Важно:

****pretraining validation loss нельзя напрямую сравнивать с SFT validation loss.****

Это разные данные и разные режимы supervision.

---

# 13. RAG простыми словами

RAG = Retrieval-Augmented Generation.

Упрощённая схема:

```text

вопрос пользователя

       ↓

embedding

       ↓

поиск похожих документов

       ↓

найденный контекст

       ↓

ответ

```

В NOVA_v1 для embeddings используется:

```text

intfloat/multilingual-e5-small

```

Размер embedding:

```text

384

```

Для поиска используется FAISS.

---

# 14. Что такое embedding

Embedding — это числовой вектор, который представляет текст.

Условно:

```text

"Что такое PostgreSQL?"

        ↓

[0.12, -0.84, 0.31, ...]

```

Похожие по смыслу тексты должны иметь близкие vector representations.

Это позволяет искать не только точное совпадение слов, но и смысловую близость.

---

# 15. Что такое FAISS

FAISS — библиотека для быстрого поиска похожих vectors.

Она не является обычной SQL-базой.

Упрощённо:

```text

тексты

  ↓

embeddings

  ↓

FAISS index

```

Для нового вопроса:

```text

вопрос

  ↓

embedding

  ↓

FAISS search

  ↓

наиболее похожие vectors

  ↓

соответствующие тексты

```

В NOVA_v1 используется inner-product search по нормализованным embeddings.

---

# 16. RAG-индексы NOVA_v1

В проекте создавались несколько экспериментальных индексов:

```text

knowledge_index

knowledge_index_v2

technical_index

core_index

question_router

trusted_index

```

Binary `.faiss` файлы являются generated artifacts и не предназначены для обычного GitHub repository.

---

# 17. Broad RAG

Большой knowledge index содержит примерно:

```text

77 900 chunks

```

Вторая версия broad-index улучшала способ построения embeddings.

Но broad RAG остался экспериментальным.

Главная проблема:

```text

retriever находит хороший контекст

              ↓

NOVA всё равно может

сгенерировать неправильный ответ

```

---

# 18. Technical RAG

Technical dataset:

```text

1 398 Q/A records

```

Темы включают:

\- Python;

\- Git;

\- Docker;

\- HTTP;

\- API;

\- SQL;

\- базы данных;

\- ML;

\- debugging;

\- безопасную работу с файлами и командами;

\- другие технические вопросы.

Technical retrieval в NOVA_v1 оставлен ручным режимом.

---

# 19. CORE knowledge

CORE — маленькая curated база доверенных ответов.

Размер:

```text

12 records

```

CORE index строится по:

```text

canonical question

\+

aliases

```

Ответ не входит в embedding.

То есть embedding нужен, чтобы найти нужный вопрос, после чего система может вернуть заранее проверенный ответ.

---

# 20. SafeRetriever

В NOVA_v1 принят консервативный режим retrieval.

Логика AUTO:

```text

AUTO

  |

  +--> CORE search

          |

          +--> score >= 0.83

          |       |

          |       +--> trusted CORE answer

          |

          +--> score < 0.83

                  |

                  +--> model fallback

```

Другие режимы:

```text

TECHNICAL -> manual

BROAD     -> manual experimental

OFF       -> no RAG

```

Автоматический каскад:

```text

CORE -> TECHNICAL -> BROAD

```

не включён намеренно.

Причина — недостаточно надёжная маршрутизация.

---

# 21. Почему появился direct-answer path

Первый вариант RAG работал классически:

```text

retrieval

   ↓

context

   ↓

prompt

   ↓

NOVA generation

```

Но тесты показали:

даже если retrieval вернул правильный факт, модель может:

\- проигнорировать контекст;

\- перепутать факты;

\- переписать ответ неправильно;

\- начать галлюцинировать.

Поэтому для trusted sources появился путь:

```text

trusted retrieval

      ↓

return stored answer directly

```

То есть модель не переписывает уже проверенный ответ.

Это один из наиболее успешных архитектурных результатов NOVA_v1.

---

# 22. Gradio UI

`app.py` предоставляет локальный интерфейс.

Основные режимы:

```text

Auto — безопасный

Technical — technical Q/A

Broad — экспериментальный

No RAG — только модель

```

UI также показывает diagnostic information и позволяет менять generation parameters.

Важно:

NOVA_v1 UI — single-turn интерфейс.

Полноценной долговременной памяти или настоящего multi-turn context пока нет.

---

# 23. No-RAG quality audit

Для проверки именно способностей FINAL checkpoint был создан тест без RAG.

Количество вопросов:

```text

27

```

Generation settings:

```text

temperature        = 0

top_k              = OFF

top_p              = OFF

repetition penalty = 1

max_new_tokens     = 160

```

Ручная строгая классификация:

```text

normal  = 1

partial = 5

wrong   = 17

garbage = 4

```

Проблемы наблюдались на:

\- арифметике;

\- географии;

\- Python;

\- TCP/UDP;

\- REST;

\- Docker;

\- Git;

\- PostgreSQL;

\- SQL;

\- database concepts.

Один из более удачных участков — pandas / missing values.

Отчёт:

```text

reports/final_quality_audit.txt

```

---

# 24. Decoding comparison

Была проверена гипотеза:

> Может быть модель знает ответы, но generation settings выбраны плохо?

Были протестированы:

```text

greedy

sampling seed 1337

sampling seed 2026

```

Менялись:

\- temperature;

\- top-k;

\- top-p;

\- repetition penalty.

Систематические ошибки остались.

Вывод:

****decoding settings не являются основной причиной проблем.****

Отчёт:

```text

reports/decoding_comparison.txt

```

---

# 25. Technical SFT

SFT = Supervised Fine-Tuning.

Цель — обучить pretrained модель отвечать как assistant.

Technical dataset:

```text

total = 1398

train = 1258

val   = 140

```

Prompt:

```text

Пользователь: \<question>

Нова:

```

Loss считается только по assistant-части.

Prompt tokens маскируются:

```text

ignore_index = -100

```

EOS остаётся supervised.

---

# 26. Почему causal shift делается в dataset

`model.forward()` не делает автоматический causal shift.

Поэтому SFT dataset строит:

```text

sequence = prompt + answer + EOS

input_ids = sequence[:-1]

labels    = sequence[1:]

```

Prompt labels заменяются на:

```text

-100

```

Таким образом задача модели:

```text

по текущему контексту

предсказывать следующий assistant token

```

---

# 27. Проверка SFT dataset

Перед обучением был выполнен полный dataset test.

Результат:

```text

TRAIN records = 1258

VAL records   = 140

TRAIN truncated = 0

VAL truncated   = 0

TRAIN RESULT = PASS

VAL RESULT   = PASS

```

Среднее количество supervised tokens:

```text

TRAIN ≈ 209.64

VAL   ≈ 197.90

```

Отчёт:

```text

reports/technical_sft_dataset_test.txt

```

---

# 28. SFT — 20-step pilot

Untouched FINAL:

```text

SFT val loss = 1.561062

```

После 20 optimizer steps:

```text

SFT val loss = 1.292469

```

Это подтвердило:

\- SFT pipeline работает;

\- gradients конечные;

\- BF16 работает;

\- checkpoint сохраняется;

\- loss уменьшается.

Но свободная generation всё ещё была плохой.

---

# 29. SFT — 120 steps

120-step run снова стартовал с исходного FINAL checkpoint.

Validation trajectory:

```text

baseline  = 1.561062

step 20   = 1.292469

step 40   = 1.201218

step 60   = 1.161714

step 80   = 1.134289

step 100  = 1.113712

step 120  = 1.096816

```

Loss стабильно уменьшался.

Gradients оставались нормальными.

То есть SFT-тренер не был сломан.

---

# 30. Но свободная generation не исправилась

После 120 steps модель всё ещё ошибалась на:

\- `is` vs `==`;

\- closure;

\- Docker stop vs kill;

\- COUNT / AVG;

\- простой арифметике;

\- PostgreSQL;

\- TCP/UDP;

\- REST;

\- других вопросах.

После этого был проведён главный финальный диагностический тест.

---

# 31. Teacher forcing и autoregressive generation

Это важно понимать.

## Teacher forcing

Модель предсказывает следующий token, но предыдущие tokens в контексте являются ****правильными эталонными tokens****.

То есть ошибка модели не портит последующий контекст.

## Autoregressive generation

Модель получает только prompt.

После этого каждый следующий token строится уже на основании ****собственных предыдущих predictions****.

Если модель ошиблась:

```text

правильный контекст

      ↓

ошибочный token

      ↓

контекст уже отличается

      ↓

следующая ошибка становится вероятнее

```

---

# 32. FINAL — teacher forcing

TRAIN:

```text

examples             = 1258

supervised tokens    = 263724

token-weighted loss  = 1.611952

perplexity           = 5.0126

token accuracy       = 64.51%

first-token accuracy = 63.83%

whole-answer exact   = 0.00%

```

VAL:

```text

examples             = 140

supervised tokens    = 27706

token-weighted loss  = 1.557662

perplexity           = 4.7477

token accuracy       = 65.01%

first-token accuracy = 65.00%

whole-answer exact   = 0.00%

```

---

# 33. SFT-120 — teacher forcing

TRAIN:

```text

loss                 = 1.082352

perplexity           = 2.9516

token accuracy       = 70.80%

first-token accuracy = 67.65%

whole-answer exact   = 0.00%

```

VAL:

```text

loss                 = 1.100093

perplexity           = 3.0044

token accuracy       = 69.87%

first-token accuracy = 65.00%

whole-answer exact   = 0.00%

```

---

# 34. Что реально улучшил SFT

TRAIN:

```text

loss

1.611952 -> 1.082352

token accuracy

64.51% -> 70.80%

first-token accuracy

63.83% -> 67.65%

```

VAL:

```text

loss

1.557662 -> 1.100093

token accuracy

65.01% -> 69.87%

first-token accuracy

65.00% -> 65.00%

```

То есть SFT действительно изменил модель и повысил вероятность правильных target tokens.

Но этого оказалось недостаточно для стабильной свободной generation.

---

# 35. Autoregressive rollout на exact TRAIN examples

Были взяты реальные TRAIN-вопросы, которые модель видела во время SFT.

Даже на них SFT-120 быстро уходила с правильной траектории.

Длина правильного token-prefix для восьми sample:

```text

1

11

2

5

1

2

1

1

```

Пример:

```text

правильный ответ:

Причина: ...

модель:

Причина: Причина: Причина: Причина: ...

```

Другой пример:

```text

задача:

написать clamp()

модель правильно начинает:

```python

но затем пишет:

def add(x, y):

    return x + y

```

Полный отчёт:

```text

reports/sft_teacher_forcing_diagnostic.txt

```

\---

\# 36. Итоговый диагноз NOVA_v1

Экспериментально подтверждено:

```text

SFT pipeline работает

        ↓

teacher-forced loss уменьшается

        ↓

teacher-forced token accuracy растёт

        ↓

НО

        ↓

autoregessive generation

остаётся нестабильной

```

Тесты не доказывают одну конкретную первопричину.

Вероятные факторы:

\- качество pretraining corpus;

\- объём качественных pretraining данных;

\- GPT-2 tokenizer для русского;

\- capacity модели;

\- context 512;

\- раннее смешение разных типов данных;

\- слишком большая задача для узкого technical SFT.

\---

\# 37. Что нельзя честно утверждать

Нельзя утверждать без дополнительных экспериментов:

```text

"во всём виноват tokenizer"

```

```text

"124M всегда слишком мало"

```

```text

"technical dataset плохой"

```

```text

"SFT не работает"

```

Корректный вывод:

\> текущая комбинация base pretraining, tokenizer, model capacity, corpus и SFT strategy не дала надёжного standalone assistant behavior.

\---

\# 38. Почему SFT не продолжается до 500/1000 steps

Потому что 120-step experiment уже показал:

\- стабильное обучение;

\- существенное улучшение teacher-forced metrics;

\- отсутствие сопоставимого улучшения free generation.

Продолжать увеличивать число шагов без изменения фундаментальных условий означало бы обучать вслепую.

Поэтому NOVA_v1 заморожена.

\---

\# 39. Что будет изменено в NOVA_v2

Основные направления:

```text

1. новый анализ корпуса;

2. строгая очистка данных;

3. отдельный tokenizer для RU + EN + code;

4. более длинный context;

5. размер модели выбирается после оценки корпуса;

6. pretraining и SFT строго разделены;

7. base model оценивается ДО SFT;

8. general assistant SFT;

9. technical SFT;

10. personality SFT;

11. отдельная Memory;

12. RAG;

13. Tools / Agent;

14. Voice;

15. benchmark NOVA_v1 vs NOVA_v2.

```

\---

\# 40. Структура проекта

Основные части:

```text

NOVA_v1/

│

├── app.py

├── README.md

├── NOVA_v1_FINAL_REPORT.md

├── requirements.txt

├── .gitignore

│

├── config/

│   ├── runtime_config.py

│   └── training/

│

├── model/

│   ├── attention.py

│   └── model.py

│

├── inference/

│   ├── assistant.py

│   └── chat.py

│

├── rag/

│   ├── embedder.py

│   ├── vector_store.py

│   ├── retriever.py

│   ├── reranker.py

│   └── safe_retriever.py

│

├── train/

│   ├── dataset.py

│   ├── trainer.py

│   ├── technical_dataset.py

│   └── technical_trainer.py

│

├── data/

│   ├── pretrain_final/

│   ├── knowledge/

│   └── nova_technical_v1_1/

│

├── scripts/

├── tests/

├── reports/

├── out/

└── archive/

```

Точное дерево проекта на момент заморозки:

```text

reports/nova_v1_final_tree.txt

```

\---

\# 41. Что читать в коде новичку

Рекомендуемый порядок.

\## Шаг 1

```text

README.md

```

Сначала понять общую систему.

\## Шаг 2

```text

model/model.py

```

Здесь находится сама языковая модель.

\## Шаг 3

```text

model/attention.py

```

Здесь attention.

\## Шаг 4

```text

inference/assistant.py

```

Здесь checkpoint превращается в работающий assistant.

\## Шаг 5

```text

rag/embedder.py

rag/vector_store.py

rag/safe_retriever.py

```

Здесь retrieval.

\## Шаг 6

```text

train/technical_dataset.py

train/technical_trainer.py

```

Здесь assistant-only SFT.

\## Шаг 7

```text

tests/

reports/

```

Здесь видно, как проверялись гипотезы и почему принимались решения.

\---

\# 42. Как читать repository другой AI-модели

Если проект анализирует другая AI-система, полезный порядок:

```text

1. README.md

2. NOVA_v1_FINAL_REPORT.md

3. model/model.py

4. model/attention.py

5. inference/assistant.py

6. rag/safe_retriever.py

7. train/technical_dataset.py

8. train/technical_trainer.py

9. reports/sft_teacher_forcing_diagnostic.txt

10. reports/final_quality_audit.txt

```

Важно сообщить AI:

\- NOVA_v1 заморожена;

\- SFT checkpoint не заменяет FINAL;

\- старые абсолютные пути в reports являются историческими;

\- большие weights/datasets намеренно отсутствуют в GitHub;

\- дальнейшее развитие идёт в NOVA_v2.

\---

\# 43. Установка окружения

\## Windows PowerShell

Перейти в проект:

```powershell

cd G:\NOVA\NOVA_v1

```

Создать новое virtual environment:

```powershell

python -m venv .venv

```

Активировать:

```powershell

.\.venv\Scripts\Activate.ps1

```

Обновить pip:

```powershell

python -m pip install --upgrade pip

```

Установить зависимости:

```powershell

pip install -r requirements.txt

```

Проверить:

```powershell

pip check

```

Важно: \`.venv\` не переносится через GitHub.

На каждом компьютере его нужно создавать заново.

\---

\# 44. После переименования проекта

Изначально clean-project назывался:

```text

NOVA_project_NEW

```

После завершения экспериментов:

```text

NOVA_v1

```

Поэтому старые reports/checkpoint metadata могут содержать:

```text

G:\NOVA\NOVA_project_NEW

```

Это \*\***не ошибка**\*\*.

Это исторический абсолютный путь на момент запуска соответствующего эксперимента.

Старые reports специально не переписываются задним числом.

\---

\# 45. Где взять checkpoint после git clone

GitHub repository по умолчанию не содержит \`.pt\` weights.

Для локального inference нужно отдельно поместить FINAL checkpoint:

```text

out/final/best_final.pt

```

Затем проверить SHA-256:

```powershell

Get-FileHash .\out\final\best_final.pt -Algorithm SHA256

```

Ожидаемый hash:

```text

765E3B02A8DA32E30D728BBE6A68F7B5130A5FA6726B0BA5775F22352D83231E

```

Если hash отличается, это уже другой файл.

\---

\# 46. Запуск приложения

После установки dependencies и размещения checkpoint:

```powershell

python .\app.py

```

Локальный адрес:

```text

http://127.0.0.1:7860

```

\---

\# 47. Примеры запуска тестов

Quality audit:

```powershell

python .\tests\test_final_quality_audit.py

```

Decoding comparison:

```powershell

python .\tests\test_decoding_comparison.py

```

SafeRetriever:

```powershell

python .\tests\test_safe_retriever.py

```

Grounded answer path:

```powershell

python .\tests\test_grounded_answer_path.py

```

Teacher-forcing diagnostic:

```powershell

python .\tests\test_sft_teacher_forcing_diagnostic.py

```

Часть тестов требует локальные:

\- checkpoints;

\- datasets;

\- FAISS indices.

Если они исключены из GitHub, соответствующий тест без них закономерно не запустится.

\---

\# 48. Reports

\`reports/\` — важная часть repository.

Это не временные logs.

Reports сохраняют экспериментальную историю:

\- что проверялось;

\- какими параметрами;

\- что получилось;

\- почему было принято следующее решение.

Особенно важны:

```text

final_quality_audit.txt

decoding_comparison.txt

technical_sft_dataset_test.txt

technical_sft_training.txt

sft_candidate_comparison.txt

sft_teacher_forcing_diagnostic.txt

core_rag_build_report.txt

core_rag_test.txt

core_gate_test.txt

technical_rag_build_report.txt

technical_rag_test.txt

broad_vs_technical_rag.txt

safe_retriever_test.txt

assistant_safe_rag_test.txt

rag_prompt_modes_test.txt

grounded_answer_path_test.txt

```

Некоторые старые reports могут быть на английском.

Они сохранены без переписывания, чтобы не изменять исторические результаты экспериментов.

Новая документация проекта ведётся на русском.

\---

\# 49. Мини-словарь

\## Token

Часть текста, с которой работает LLM.

Один token не обязательно равен одному слову.

\## Tokenizer

Алгоритм:

```text

текст -> token IDs

```

и обратно.

\## BPE

Byte Pair Encoding.

Один из способов построения tokenizer.

\## Checkpoint

Сохранённое состояние модели.

Обычно содержит weights и training metadata.

\## Weights

Числовые параметры neural network.

Именно они изменяются во время обучения.

\## Pretraining

Базовое обучение language model предсказывать следующий token на большом corpus.

\## SFT

Supervised Fine-Tuning.

Дообучение pretrained модели на примерах:

```text

instruction / question

        ↓

desired answer

```

\## Teacher forcing

Во время обучения/оценки модель получает правильные предыдущие target tokens.

\## Autoregressive generation

Модель получает prompt, а затем использует собственные предыдущие predictions для генерации следующих tokens.

\## Embedding

Числовой vector, представляющий текст.

\## RAG

Retrieval-Augmented Generation.

Система ищет внешние знания перед формированием ответа.

\## FAISS

Библиотека для быстрого поиска похожих vectors.

\## Reranker

Дополнительная модель, которая может переупорядочить найденные retrieval results.

\## Perplexity

Метрика language modeling.

Обычно меньшая perplexity лучше в рамках одного и того же evaluation setup.

Нельзя бездумно сравнивать perplexity на разных datasets/objectives.

\---

\# 50. Безопасность перед публикацией

Никогда не публиковать:

```text

.env

API keys

access tokens

passwords

private keys

личную память

приватные документы

секретные конфиги

```

Если секрет случайно попал в публичный Git repository, недостаточно просто удалить файл следующим commit.

Нужно:

```text

1. немедленно отозвать/rotate secret;

2. затем очищать Git history.

```

\---

\# 51. Важный нюанс \`.gitignore\`

\`.gitignore\` не удаляет уже tracked files.

Он только говорит Git:

\> новые подходящие файлы не нужно автоматически добавлять.

Поэтому перед первым push необходимо проверить:

```powershell

git status

```

И после \`git add\` обязательно:

```powershell

git diff --cached --name-only

```

Эта команда показывает, что именно попадёт в следующий commit.

\---

\# 52. Почему reports остаются в GitHub

Reports относительно маленькие и позволяют восстановить логику проекта.

По ним другой разработчик или AI может увидеть:

\- реальные результаты;

\- неудачные эксперименты;

\- успешные эксперименты;

\- параметры тестов;

\- причину появления NOVA_v2.

Поэтому текстовые reports являются частью проекта.

\---

\# 53. Что сознательно НЕ переносится напрямую в NOVA_v2

NOVA_v2 не создаётся по схеме:

```text

copy NOVA_v1

rename

continue patching

```

Она начинается как новый проект.

Но отдельные проверенные идеи могут быть перенесены:

\- checkpoint validation;

\- atomic saving;

\- evaluation methodology;

\- SafeRetriever concept;

\- структура reports;

\- подход к тестированию.

Каждый переносимый компонент должен заново проверяться в v2.

\---

\# 54. Что считать успехом NOVA_v1

Успех NOVA_v1 не означает:

```text

"получился аналог ChatGPT"

```

Результат версии:

```text

custom Transformer

\+

собственный pretraining

\+

реальные checkpoints

\+

local inference

\+

RAG

\+

SFT

\+

evaluation

\+

диагностика failures

\+

понятные требования к v2

```

Это законченная первая исследовательская итерация.

\---

\# 55. Финальный статус

```text

NOVA_v1

\========

Custom LLM                     YES

Custom GPT implementation      YES

Training from scratch          YES

Local GPU inference            YES

CUDA / BF16                    YES

RAG                            YES

SafeRetriever                  YES

Gradio UI                      YES

Assistant-only SFT             YES

Evaluation suite               YES

Reliable standalone assistant  NO

Reliable generative grounding  NO

Stable SFT free generation     NO

Development                    FROZEN

Role                           BASELINE / EXPERIMENT

Next                           NOVA_v2

```

\---

\# 56. NOVA_v2

Перед написанием основной модели NOVA_v2 будет создан отдельный design document:

```text

NOVA_v2_DESIGN.md

```

В нём должны быть заранее определены:

\- цель модели;

\- corpus;

\- tokenizer;

\- architecture;

\- context length;

\- parameter count;

\- training budget;

\- throughput;

\- evaluation;

\- assistant SFT;

\- personality;

\- memory;

\- RAG;

\- tools;

\- voice.

Главное правило второй версии:

\> сначала данные, архитектурное обоснование и критерии проверки — потом длинное обучение.

\---

\# 57. Заключение

NOVA_v1 — первая законченная попытка построить NOVA как собственную локальную LLM.

Она не достигла конечной цели универсального персонального AI-ассистента.

Но дала важный результат:

\- рабочий ML pipeline;

\- работающую custom architecture;

\- набор воспроизводимых экспериментов;

\- чёткие отрицательные результаты;

\- понимание границ v1;

\- требования к следующей версии.

Поэтому NOVA_v1 не удаляется и не переписывается задним числом.

Она остаётся неизменным baseline.

Дальнейшее развитие происходит в:

```text

NOVA_v2

```
