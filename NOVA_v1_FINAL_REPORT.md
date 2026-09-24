# NOVA_v1 — ФИНАЛЬНЫЙ ОТЧЁТ

**Проект:** NOVA  
**Версия:** NOVA_v1  
**Статус:** АРХИВ / ТОЛЬКО ДЛЯ ЧТЕНИЯ  
**Дата фиксации версии:** 24 сентября 2026  
**Назначение документа:** сохранить техническое состояние, эксперименты, результаты, ограничения и выводы NOVA_v1 перед началом разработки NOVA_v2.

---

## 1. Краткий итог

NOVA_v1 — первая законченная экспериментальная версия собственной локальной decoder-only языковой модели и системы AI-ассистента.

В рамках проекта были реализованы и проверены:

- собственный GPT-подобный decoder-only Transformer;
- локальная загрузка checkpoint и генерация текста;
- GPT-2 byte-level BPE токенизация через `tiktoken`;
- pretraining checkpoint приблизительно на 124 млн обучаемых параметров;
- генерация без RAG;
- несколько RAG-индексов на базе multilingual E5 embeddings и FAISS;
- маршрутизация через SafeRetriever;
- прямой возврат доверенного ответа для выбранных RAG-источников;
- приложение на Gradio;
- technical SFT pipeline с обучением только на assistant-части;
- наборы тестов для качества модели, decoding, RAG, SFT и сравнения teacher forcing с autoregressive generation.

NOVA_v1 **не считается надёжным самостоятельным ассистентом**.

Главный экспериментальный результат версии: technical SFT действительно уменьшал teacher-forced loss и повышал teacher-forced token accuracy, однако это улучшение не превратилось в устойчивую autoregressive generation. Даже на точных примерах из обучающей выборки SFT-модель обычно сходила с эталонной траектории уже через несколько токенов.

Поэтому NOVA_v1 заморожена как экспериментальный baseline. Дальнейшая разработка продолжается в новом проекте `NOVA_v2`, где планируется заново спроектировать фундамент модели: улучшить pretraining data, использовать tokenizer, подходящий для русского + английского + кода, осознанно выбрать архитектуру, увеличить context length и строго разделить pretraining, assistant SFT, personality, RAG, memory и tools.

---

## 2. Исходная цель

Долгосрочная цель NOVA — создать локального персонального AI-ассистента на основе собственной модели, разработанной и обученной с нуля, а не просто переименовать или слегка дообучить готовую pretrained-модель.

В дальнейшем NOVA должна сочетать:

- естественное общение на русском языке;
- английский язык при необходимости;
- устойчивый характер и стиль общения;
- честное признание неопределённости и нехватки знаний;
- способность поддерживать обычный диалог;
- сильную специализацию в Python, AI и разработке;
- внешние знания через RAG;
- долговременную память вне весов модели;
- инструменты для работы с файлами, кодом и локальной средой;
- голосовой ввод и вывод.

NOVA_v1 стала первой попыткой построить модель и окружающую систему, необходимые для этой цели.

---

## 3. Финальное имя и расположение проекта

Чисто пересобранный проект изначально находился по пути:

```text
G:\NOVA\NOVA_project_NEW
```

После завершения финальной диагностики проект был переименован в:

```text
G:\NOVA\NOVA_v1
```

Некоторые старые отчёты и metadata checkpoint, созданные до переименования, всё ещё содержат старый абсолютный путь `G:\NOVA\NOVA_project_NEW`.

Это историческая информация, а не признак существования второй активной версии проекта.

Два старых проекта, из которых при пересборке выбирались отдельные компоненты, больше не являются рабочими ветками. NOVA_v1 считается сохранённым baseline.

---

## 4. Аппаратная и программная среда

### Аппаратная часть

- CPU: Intel Core i9-14900KS
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- VRAM: 16 GB
- RAM: 96 GB DDR5
- Материнская плата: MSI MAG Z790 Tomahawk WiFi
- Основной SSD: Samsung 990 Pro M.2 2 TB
- ОС: Windows 11 Pro 24H2

### Python-окружение NOVA_v1

Использовалось локальное виртуальное окружение `.venv`.

Основные версии пакетов:

```text
torch==2.14.0+cu130
numpy==2.5.3
tiktoken==0.14.0
tqdm==4.70.1
transformers==5.17.0
tokenizers==0.23.2
huggingface-hub==1.32.0
sentencepiece==0.2.2
safetensors==0.8.0
faiss-cpu==1.15.1
gradio==6.28.0
```

PyTorch корректно определял CUDA, видел RTX 4070 Ti SUPER и подтверждал поддержку BF16.

---

## 5. Архитектура модели NOVA_v1

Финальный baseline checkpoint использует собственный GPT-подобный decoder-only Transformer.

### Основные параметры

```text
vocab_size:      50 257
embedding size:  768
attention heads: 12
layers:          12
context:         512 tokens
dropout:         0.1 во время обучения
weight tying:    YES
parameters:      124 009 728 уникальных обучаемых параметров
```

### Детали Transformer

Реализация использует:

- token embeddings;
- learned absolute position embeddings;
- pre-norm Transformer blocks;
- LayerNorm;
- causal self-attention;
- PyTorch scaled dot-product attention;
- feed-forward network с расширением hidden-размера в 4 раза;
- GELU;
- residual connections;
- scaled residual initialization;
- общую матрицу весов для token embedding и output LM head.

Модель является autoregressive next-token language model.

### Поведение loss

`model.forward()` не выполняет causal shift автоматически.

Следовательно, для pretraining и SFT входы и targets должны быть заранее правильно сдвинуты.

Для technical assistant SFT dataset явно формирует:

```text
input  = sequence[:-1]
target = sequence[1:]
```

а prompt tokens маскируются значением:

```text
ignore_index = -100
```

Эта логика отдельно проверялась перед запуском SFT.

---

## 6. Tokenizer

NOVA_v1 использует:

```python
tiktoken.get_encoding("gpt2")
```

Это GPT-2 byte-level BPE с размером словаря 50 257 tokens.

### Плюсы решения в NOVA_v1

- простая и стабильная готовая реализация;
- отсутствие классической проблемы неизвестного слова;
- возможность кодировать любой Unicode-текст через byte-level представление;
- совместимость с существующей GPT-подобной архитектурой.

### Обнаруженное ограничение

GPT-2 BPE не создавался специально для русского языка.

Русский текст может разбиваться на сравнительно неэффективные byte/subword последовательности. Это может увеличивать длину последовательностей и усложнять небольшой модели изучение русской морфологии и структуры слов.

Это **гипотеза для NOVA_v2**, а не доказанная единственная причина проблем NOVA_v1.

Поэтому в NOVA_v2 планируется исследовать собственный tokenizer, обученный на смеси:

- русского языка;
- английского языка;
- Python и другого кода;
- технической терминологии.

---

## 7. Pretraining data

Фактическое состояние файлов на момент архивации NOVA_v1:

```text
data/
├── pretrain_final/
│   └── train.txt
└── val.txt
```

Размеры:

```text
train corpus:
116 602 874 bytes

validation corpus:
18 008 668 bytes
```

SHA-256 файла `data/val.txt`:

```text
25BED3A3C6543EBBB94B42AD9297FC9095A48C89FC7CAD61B4E9EDDE2F18F692
```

Train corpus соответствовал приблизительно 66.6 млн GPT-2 tokens.

Ранее в рабочей документации validation corpus описывался как:

```text
data/pretrain_final/val.txt
```

Однако в финальном архиве NOVA_v1 такого файла нет.

Фактически validation corpus расположен в:

```text
data/val.txt
```

При дополнительной проверке установлено, что:

```text
config/training/pretrain_config.py
scripts/prepare_pretrain_data.py
```

существуют, но имеют размер `0 bytes`.

Следовательно, по сохранённому состоянию проекта невозможно достоверно восстановить, каким именно способом пути к train и validation corpus передавались во время первоначального pretraining run.

Этот отчёт намеренно не заполняет данный пробел предположениями.

Финальный корпус был собран из нескольких более ранних источников, включая general, synthetic и instruction-style материалы.

### Важный вывод

На ранних этапах NOVA_v1 граница между следующими типами данных была недостаточно строгой:

- pretraining corpus;
- instruction data;
- assistant data;
- synthetic Q/A data.

В NOVA_v2 эти роли должны быть разделены явно ещё до начала обучения.

---

## 8. Финальная конфигурация pretraining

Примерная финальная конфигурация:

```text
batch_size:                  4
gradient_accumulation_steps: 8
effective batch:             32 sequences
learning_rate:               1e-4
min_lr:                      1e-5
weight_decay:                0.1
betas:                       (0.9, 0.95)
grad_clip:                   1.0
max_iters:                   50 000
eval_interval:               1 000
eval_iters:                  50
save_interval:               2 500
dtype:                       bfloat16
device:                      cuda
context/block_size:          512
```

Сохранённый FINAL checkpoint соответствует:

```text
iter_num:      45 000
best_val_loss: 0.515568
```

Pretraining validation loss нельзя напрямую сравнивать с SFT validation loss, потому что используются разные datasets и разные supervision objectives.

---

## 9. Сохранённые checkpoints

### FINAL pretraining checkpoint

Относительный путь:

```text
out/final/best_final.pt
```

SHA-256:

```text
765E3B02A8DA32E30D728BBE6A68F7B5130A5FA6726B0BA5775F22352D83231E
```

Metadata:

```text
iter_num:      45000
best_val_loss: 0.515568
architecture:  tied weights
parameters:    124.010M
```

### SFT 20-step checkpoint

Относительный путь:

```text
out/candidate_sft/technical_v1_step20/best_candidate.pt
```

SHA-256:

```text
19B88F804126B085AE70830E2654589EED2899E76DE65804968CE331302EDE90
```

Этот checkpoint сохранён как экспериментальная контрольная точка и не является основной моделью.

### SFT 120-step checkpoint

Относительный путь:

```text
out/candidate_sft/technical_v1_step120/best_candidate.pt
```

SHA-256:

```text
3C1C3D89C81B16EDF6C982210BEDB33B04512FE25B7DF9BBFF6BFC10DDDD36A8
```

Metadata:

```text
iter_num:      45120
SFT val loss:  приблизительно 1.0968 в trainer evaluation
architecture:  tied weights
```

Этот checkpoint также сохранён только как экспериментальный артефакт и не заменяет FINAL.

---

## 10. Архитектура RAG

NOVA_v1 содержит несколько RAG-экспериментов.

### Embedding model

Основная embedding-модель:

```text
intfloat/multilingual-e5-small
```

Размер embedding:

```text
384
```

Новые индексы используют attention-mask-aware mean pooling (`masked_mean`) с последующей нормализацией vectors.

### Vector search

Для similarity search используется FAISS.

Основной vector store использует inner-product search по нормализованным векторам, что по смыслу эквивалентно cosine-like ranking.

### Broad RAG

Большой knowledge index содержит приблизительно:

```text
77 900 chunks
```

Вторая версия broad index (`knowledge_index_v2`) строилась с улучшенным способом формирования embeddings по сравнению с legacy index.

Broad RAG остался экспериментальным, потому что retrieval мог вернуть релевантный контекст, а маленький generator всё равно использовал его ненадёжно.

### Technical RAG

Technical corpus содержит:

```text
1 398 question/answer records
```

Technical FAISS index использует multilingual E5 embeddings.

Technical RAG доступен как ручной режим retrieval.

### CORE RAG

Маленькая curated CORE knowledge base содержит:

```text
12 trusted records
```

Каждая запись включает canonical question, aliases и короткий доверенный ответ.

CORE index строится по вопросу и aliases, но не по самому ответу.

Порог принятия:

```text
0.83
```

был выбран как предварительный по ограниченному gate test.

В этом ограниченном тесте:

```text
известные CORE positives приняты: 12/12
negative/OOD примеры отклонены:    12/12
```

Этот threshold не считается глобально доказанным.

### SafeRetriever

Принятая схема маршрутизации NOVA_v1:

```text
AUTO
    -> только CORE
    -> если top-1 score >= threshold:
           принять один trusted CORE result
       иначе:
           не использовать RAG и перейти к model fallback

TECHNICAL
    -> ручной режим

BROAD
    -> ручной экспериментальный режим

OFF
    -> только модель
```

Автоматический fallback CORE -> TECHNICAL -> BROAD намеренно не включался.

---

## 11. Direct trusted-answer path

Ключевой вывод RAG-экспериментов: корректно найденный контекст не гарантирует корректную генерацию.

В тестах встречались ситуации, когда retriever находил правильный контекст по PostgreSQL, REST или Python, а FINAL-модель всё равно генерировала неправильный ответ.

Проверялись разные prompt modes, однако одно только изменение prompt format не обеспечило надёжное grounding.

Поэтому в NOVA_v1 был принят обходной путь:

```text
trusted CORE / выбранный TECHNICAL result
              |
              v
       вернуть trusted answer напрямую,
       не заставляя LLM переписывать его
```

Для untrusted или broad retrieval генерация осталась экспериментальной.

Это одно из наиболее удачных архитектурных решений NOVA_v1.

---

## 12. Gradio-приложение

`app.py` предоставляет локальный интерфейс с четырьмя режимами:

```text
Auto — безопасный
Technical — trusted technical Q/A
Broad — экспериментальный
No RAG — только модель
```

UI также содержит generation controls и diagnostic information.

Важное ограничение: интерфейс NOVA_v1 является single-turn.

Предыдущие UI-сообщения не формируют полноценную multi-turn conversation history для модели.

Для NOVA_v1 это допустимо, но не является целевой архитектурой памяти.

---

## 13. Аудит качества standalone-модели

Для FINAL checkpoint был проведён no-RAG audit на 27 вопросах с deterministic greedy decoding:

```text
temperature:        0
top_k:              off
top_p:              off
repetition penalty: 1
max_new_tokens:     160
```

Строгая ручная классификация:

```text
normal:   1
partial:  5
wrong:   17
garbage:  4
```

Наблюдались ошибки в:

- простой арифметике;
- столице Австралии;
- базовых Python-концепциях;
- `is` vs `==`;
- mutable default arguments;
- TCP vs UDP;
- REST;
- PostgreSQL;
- SQL;
- Docker;
- Git.

Вопрос по pandas missing values оказался одним из наиболее сильных ответов.

### Интерпретация

Этот тест не доказывает, что модель «ничего не знает».

Он показывает, что FINAL checkpoint ненадёжен как самостоятельный ассистент на проверенном наборе базовых, общих и технических вопросов.

---

## 14. Decoding comparison

Второй эксперимент проверял 10 типовых prompt в трёх режимах:

```text
1. greedy
2. app-like sampling, seed 1337
3. app-like sampling, seed 2026
```

Систематические ошибки сохранялись во всех режимах.

Изменение temperature, top-k, top-p и repetition penalty не исправляло ошибки на:

- арифметике;
- Австралии;
- Python comparison semantics;
- TCP/UDP;
- REST;
- Docker vs VM;
- Git fetch/pull;
- PostgreSQL.

### Вывод

Decoding configuration не была основной причиной наблюдавшихся ошибок качества.

---

## 15. Technical SFT dataset

Assistant-only technical SFT dataset содержит:

```text
total: 1 398
train: 1 258
val:     140
```

Проверка dataset подтвердила:

```text
block_size:   512
EOS token:    50256
ignore_index: -100
train truncated: 0
val truncated:   0
```

Статистика supervised tokens:

```text
TRAIN
min:  37
max:  401
mean: 209.64

VAL
min:  30
max:  403
mean: 197.90
```

Формат prompt:

```text
Пользователь: <question>
Нова:
```

Loss считается только по assistant-части.

Prompt/user tokens маскируются.

Настоящий EOS token остаётся supervised.

---

## 16. SFT — 20-step pilot

Исходный FINAL checkpoint был оценён на SFT validation set:

```text
baseline val loss: 1.561062
```

После 20 optimizer steps:

```text
val loss: 1.292469
```

Pipeline показал:

- стабильное BF16 обучение;
- конечные gradients;
- уменьшение training loss;
- корректное сохранение checkpoint;
- отсутствие ошибок в data pipeline.

Однако сравнение free generation показало, что ответы на technical и general вопросы всё ещё оставались плохими.

### Вывод

SFT implementation технически работал, но 20 steps не дали надёжного autoregressive assistant behavior.

---

## 17. SFT — 120-step experiment

После этого был выполнен контролируемый 120-step run, снова начиная с untouched FINAL checkpoint.

Validation trajectory:

```text
baseline  1.561062
step 20   1.292469
step 40   1.201218
step 60   1.161714
step 80   1.134289
step 100  1.113712
step 120  1.096816
```

Gradient norm оставался конечным и в целом снижался.

Лучший validation result наблюдался на step 120.

### Важное наблюдение

Teacher-forced validation objective улучшался, но качество free generation не стало надёжным.

Даже exact/near-exact technical вопросы вроде:

- Python `is` vs `==`;
- closures;
- `docker stop` vs `docker kill`;
- SQL COUNT/AVG;

всё ещё получали неправильные autoregressive answers.

---

## 18. Финальная диагностика: teacher forcing vs autoregressive generation

Это последний эксперимент, после которого NOVA_v1 была закрыта.

Диагностика сравнила untouched FINAL checkpoint и SFT-120 checkpoint на полных technical train и validation splits.

### FINAL — teacher forcing

TRAIN:

```text
examples:             1 258
supervised tokens:    263 724
token-weighted loss:  1.611952
perplexity:           5.0126
token accuracy:       64.51%
first-token accuracy: 63.83%
whole-answer exact:   0.00%
```

VAL:

```text
examples:             140
supervised tokens:    27 706
token-weighted loss:  1.557662
perplexity:           4.7477
token accuracy:       65.01%
first-token accuracy: 65.00%
whole-answer exact:   0.00%
```

### SFT-120 — teacher forcing

TRAIN:

```text
examples:             1 258
supervised tokens:    263 724
token-weighted loss:  1.082352
perplexity:           2.9516
token accuracy:       70.80%
first-token accuracy: 67.65%
whole-answer exact:   0.00%
```

VAL:

```text
examples:             140
supervised tokens:    27 706
token-weighted loss:  1.100093
perplexity:           3.0044
token accuracy:       69.87%
first-token accuracy: 65.00%
whole-answer exact:   0.00%
```

### Прямое сравнение

TRAIN:

```text
loss:                 1.611952 -> 1.082352
token accuracy:       64.51%   -> 70.80%
first-token accuracy: 63.83%   -> 67.65%
whole-answer exact:   0.00%    -> 0.00%
```

VAL:

```text
loss:                 1.557662 -> 1.100093
token accuracy:       65.01%   -> 69.87%
first-token accuracy: 65.00%   -> 65.00%
whole-answer exact:   0.00%    -> 0.00%
```

### Greedy rollout на точных TRAIN examples

Для восьми точных обучающих примеров была выполнена autoregressive generation только из prompt.

SFT-модель очень быстро сходила с целевого ответа.

Наблюдавшиеся длины правильного token-prefix:

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

tokens для восьми выбранных train examples.

Ни один sample не прошёл весь reference answer без расхождения.

Примеры:

- в задаче про строковую переменную окружения модель правильно начинала с `Причина:`, но затем начинала повторять `Причина:` вместо завершения объяснения;
- в задаче на функцию `clamp()` модель правильно начинала Python code block, но затем генерировала нерелевантную функцию сложения;
- несколько diagnostic/safety ответов расходились с эталоном уже после одного-двух tokens.

### Интерпретация

SFT научил модель повышать вероятность многих правильных target tokens, когда правильная предыдущая история подавалась через teacher forcing.

Но модель не научилась достаточно устойчивой самостоятельной autoregressive траектории ответа.

First-token metric также нужно трактовать осторожно: GPT-2 byte-level BPE может кодировать первый элемент ответа как пробел, часть слова или byte/subword token, а не как целое смысловое слово.

---

## 19. Что в NOVA_v1 сработало

### Модель и инфраструктура

Успешно реализованы:

- custom decoder-only Transformer;
- tied-weight финальная архитектура;
- корректное восстановление модели из checkpoint;
- CUDA/BF16 inference;
- CUDA/BF16 training;
- optimizer setup;
- checkpoint save/load;
- deterministic evaluation;
- generation controls;
- локальное virtual environment;
- SHA-256 идентификация важных checkpoint-файлов.

### SFT pipeline

Assistant-only SFT pipeline был проверен:

- causal shift корректен;
- prompt masking корректен;
- EOS supervision корректен;
- padding labels замаскированы;
- truncation отсутствует;
- полный train/val dataset test проходит;
- loss уменьшается;
- validation улучшается;
- candidate checkpoints сохраняются отдельно и не перезаписывают FINAL.

### RAG

Успешные компоненты:

- multilingual E5 embeddings;
- FAISS vector indices;
- technical index;
- CORE curated index;
- SafeRetriever routing;
- консервативный CORE gate;
- direct trusted-answer path.

Особенно важным оказался direct-answer подход, поскольку он не позволял слабому generator переписывать корректный trusted fact в неправильный ответ.

### Evaluation methodology

NOVA_v1 получила собственный набор проверок вместо оценки только «на глаз» через чат.

В него вошли:

- integrity checks;
- no-RAG audit;
- decoding comparison;
- broad vs technical RAG comparison;
- CORE gate tests;
- SafeRetriever tests;
- grounded-answer tests;
- SFT dataset validation;
- 20-step SFT pilot;
- 120-step SFT experiment;
- финальная teacher-forcing/autoregressive diagnostic.

---

## 20. Что не работало надёжно

### Standalone assistant behavior

FINAL-модель ненадёжна на базовых factual, arithmetic, Python, networking, web, Docker, Git и database questions.

### Prompt engineering

Изменение RAG prompt format не заставляло модель стабильно использовать правильный retrieved context.

### Sampling/decoding

Изменение greedy/sampling параметров не исправляло систематические knowledge/instruction failures.

### Generative RAG

Retriever мог найти правильный ответ, а generator всё равно выдавал неправильный.

### Technical SFT

SFT улучшил teacher-forced metrics, но не дал надёжных autoregressive technical answers.

### Generalization

Technical dataset слишком узок, чтобы научить модель общей арифметике, географии и другим широким знаниям, если base model сама не имеет достаточно сильного общего фундамента.

---

## 21. Что NOVA_v1 НЕ доказывает

Проведённые эксперименты **не доказывают** ни одно из следующих утверждений:

```text
"Technical dataset плохой."
"Tokenizer точно является главной проблемой."
"124M параметров всегда слишком мало."
"Context 512 вызвал все ошибки."
"Дополнительное обучение вообще никогда не помогло бы."
"RAG не работает."
"SFT не работает."
```

Корректный более узкий вывод:

> Текущая комбинация base pretraining, model capacity, tokenizer, данных, context length и SFT strategy не дала надёжного standalone autoregressive assistant.

Вклад каждого отдельного фактора в NOVA_v1 изолированно не измерялся.

---

## 22. Основные гипотезы для NOVA_v2

Следующие пункты являются гипотезами, а не доказанными фактами.

### 22.1 Качество pretraining corpus

Base model могла получить недостаточно чистых, разнообразных и качественных general/technical pretraining data.

### 22.2 Масштаб pretraining corpus

Объём действительно качественного pretraining текста мог быть недостаточен для ожидаемых способностей.

### 22.3 Несоответствие tokenizer

GPT-2 byte-level BPE может быть неэффективным для русскоязычного ассистента.

### 22.4 Model capacity

124M параметров могут быть недостаточны для одновременного сочетания:

- свободного русского языка;
- общих знаний;
- технических знаний;
- instruction following;
- personality;
- code assistance.

Но размер NOVA_v2 должен определяться только после измерения реального v2 corpus.

### 22.5 Context length

512 tokens ограничивают:

- source code;
- tracebacks;
- RAG context;
- multi-turn assistant behavior;
- будущий tool use.

### 22.6 Разделение этапов обучения

В NOVA_v2 необходимо намного чётче разделять:

- pretraining;
- instruction tuning;
- technical specialization;
- personality;
- memory;
- RAG.

---

## 23. Почему создаётся NOVA_v2

NOVA_v2 создаётся не потому, что NOVA_v1 «бесполезна».

Наоборот, NOVA_v1 дала эксперименты, необходимые для понимания того, что требуется изменить.

Переход к v2 основан на следующих наблюдениях:

1. модель технически обучается и сохраняется корректно;
2. FINAL checkpoint генерирует текст, но качество самостоятельных ответов ненадёжно;
3. decoding changes не исправляют систематические failures;
4. retrieval сам по себе может работать;
5. prompt-only grounding ненадёжен;
6. direct trusted retrieval полезен;
7. assistant-only SFT технически работает;
8. SFT заметно улучшает teacher-forced metrics;
9. это улучшение не превращается в стабильную autoregressive generation;
10. даже exact train examples быстро сходят с правильной траектории при greedy rollout.

Поэтому дальнейшее слепое увеличение SFT steps не имеет достаточного экспериментального основания.

Следующая итерация должна улучшать фундамент модели, а не продолжать латать NOVA_v1.

---

## 24. Принципы NOVA_v2, полученные из опыта v1

NOVA_v2 должна начинаться как отдельный чистый проект.

Она **не должна** создаваться путём полного копирования NOVA_v1 и дальнейшего исправления поверх неё.

При этом отдельные проверенные компоненты v1 можно осознанно переносить после повторной проверки.

Планируемые принципы:

```text
1. Сначала определить требования к модели и системе.
2. Собрать и проверить corpus до выбора окончательного размера модели.
3. Обучить tokenizer, подходящий для русского + английского + кода.
4. Разделить pretraining data и instruction/personality data.
5. Увеличить context выше 512.
6. Измерить производительность архитектуры на реальной RTX 4070 Ti SUPER.
7. Проводить короткие training pilots до длинного pretraining run.
8. Оценивать base model ДО assistant SFT.
9. Не использовать SFT как замену отсутствующим base capabilities.
10. Осознанно смешивать general assistant, technical, uncertainty и personality SFT.
11. Хранить долговременную пользовательскую память вне model weights.
12. Отделять trusted knowledge/RAG от personality.
13. Добавлять tools только после стабилизации текстового assistant core.
14. Добавлять voice после стабилизации текстовой системы.
15. Сравнивать NOVA_v2 с сохранённым NOVA_v1 baseline.
```

---

## 25. Планируемое концептуальное разделение NOVA_v2

Долгосрочная система должна разделять:

```text
NOVA Core
    базовая языковая модель

NOVA Assistant
    instruction following
    assistant behavior
    technical specialization

NOVA Personality
    устойчивый стиль общения
    ценности
    тон
    поведенческая последовательность

Memory
    изменяемая долговременная информация
    история пользователя и проектов
    факты из разговоров

RAG
    внешние знания
    документация
    trusted factual sources

Tools / Agent
    файлы
    Python
    локальные действия
    внешние интеграции

Voice
    speech-to-text
    text-to-speech
```

Personality нельзя путать с memory.

Изменяемые факты о пользователе не должны постоянно «запекаться» в weights модели.

---

## 26. Сохраняемые reports и artifacts

Важные отчёты NOVA_v1 должны храниться в `reports/`.

Ключевые файлы:

```text
nova_v1_final_tree.txt
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

Точное дерево проекта на момент архивации сохранено в:

```text
reports/nova_v1_final_tree.txt
```

Дополнительно установлено, что два pretraining-related файла:

```text
config/training/pretrain_config.py
scripts/prepare_pretrain_data.py
```

имеют размер `0 bytes` и сохраняются только как часть фактического состояния архива.

Старые reports не переписываются задним числом, даже если часть из них на английском. Они являются историческими артефактами конкретных запусков.

---

## 27. Политика архива

NOVA_v1 после фиксации считается read-only baseline.

Допустимые действия в будущем:

- просматривать;
- запускать старые tests после восстановления окружения;
- сравнивать с NOVA_v2;
- брать отдельный проверенный utility для повторного использования;
- анализировать исторические результаты.

Не рекомендуется:

- продолжать SFT;
- перезаписывать FINAL;
- менять RAG thresholds и всё ещё называть результат тем же baseline;
- изменять архитектуру модели «внутри v1»;
- удалять экспериментальные checkpoint, на которые ссылается этот отчёт.

Если какой-либо компонент переносится в NOVA_v2, его нужно переносить отдельно и тестировать заново.

---

## 28. Финальный статус

```text
NOVA_v1
========

Собственная LLM:                         ДА
Собственная GPT-реализация:              ДА
Обучение с нуля:                         ДА
Локальный inference:                     ДА
CUDA/BF16:                               ДА
RAG:                                     ДА
Safe retrieval:                          ДА
Gradio UI:                               ДА
Assistant-only SFT:                      ДА
Evaluation suite:                        ДА

Надёжный standalone assistant:           НЕТ
Надёжное generative grounding:           НЕТ
Стабильная free generation после SFT:    НЕТ

Исследовательская ценность:              ВЫСОКАЯ
Production readiness:                    НЕТ
Статус разработки:                       ЗАМОРОЖЕНА
Следующая версия:                        NOVA_v2
```

---

## 29. Финальный вывод

NOVA_v1 выполнила свою главную инженерную задачу: превратила первоначальную идею в реальную, обучаемую и тестируемую локальную языковую модель и дала достаточно данных, чтобы понять ограничения первого дизайна.

Проект показал, что:

- уменьшение training/validation loss само по себе не равно качеству ассистента;
- корректный retriever не гарантирует grounded generation;
- prompt engineering не может компенсировать все слабости base model;
- SFT нельзя считать заменой отсутствующим base capabilities;
- надёжная evaluation должна включать свободную autoregressive generation, а не только teacher-forced loss.

Поэтому NOVA_v1 сохраняется как baseline и экспериментальная история проекта.

NOVA_v2 начинается с выводов v1, сохраняя ту же долгосрочную цель, но получая новый технический фундамент.
