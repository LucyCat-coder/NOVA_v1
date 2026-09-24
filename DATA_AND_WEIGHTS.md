# NOVA_v1 — данные, веса и ограничения публикации

Этот файл описывает, какие артефакты NOVA_v1 публикуются в GitHub, какие остаются только локально и почему.

NOVA_v1 публикуется как **архивная экспериментальная версия / baseline**, а не как полностью воспроизводимый пакет с весами и всеми исходными данными.

---

## 1. Что входит в GitHub-репозиторий

В публичный repository входят:

- исходный код модели, inference, RAG и SFT;
- конфигурационные файлы, сохранившиеся в архиве;
- tests;
- scripts;
- README и финальные отчёты;
- текстовые reports экспериментов;
- `data/knowledge/core_knowledge.jsonl`;
- метаданные technical dataset:
  - `data/nova_technical_v1_1/READ_ME.txt`;
  - `data/nova_technical_v1_1/report.json`.

Некоторые файлы архива могут быть пустыми placeholder-файлами. В частности:

```text
config/training/pretrain_config.py
scripts/prepare_pretrain_data.py
```

На момент архивации они имеют размер `0 bytes` и сохраняются как часть фактического состояния NOVA_v1.

---

## 2. Что намеренно НЕ публикуется

Через `.gitignore` исключены:

- `.venv/`;
- Python caches и `__pycache__/`;
- model checkpoints и другие бинарные weights;
- каталог `out/`;
- FAISS binary indices;
- generated RAG index metadata;
- большой pretraining corpus;
- локальный validation corpus;
- большой broad knowledge corpus;
- raw technical dataset;
- локальные архивы;
- caches, временные файлы и потенциальные secrets.

Отсутствие этих файлов после `git clone` является ожидаемым.

---

## 3. Pretraining corpus

Фактическое локальное состояние архива:

```text
data/
├── pretrain_final/
│   └── train.txt
└── val.txt
```

Размеры:

```text
data/pretrain_final/train.txt: 116 602 874 bytes
data/val.txt:                   18 008 668 bytes
```

SHA-256 локального validation corpus:

```text
25BED3A3C6543EBBB94B42AD9297FC9095A48C89FC7CAD61B4E9EDDE2F18F692
```

Train corpus содержал приблизительно `66.6M` GPT-2 tokens.

Оба корпуса исключены из публичного GitHub repository.

Причины:

1. это крупные производные training artifacts;
2. для части ранних источников недостаточно полно сохранена provenance/licensing документация;
3. NOVA_v1 публикуется прежде всего как инженерный и исследовательский архив, а не как redistributable training-corpus release.

---

## 4. Technical dataset

Подготовленный technical dataset имеет статус:

```text
PREPARED_WITH_LIMITED_AUDIT
```

Согласно сохранённому `report.json`:

```text
до подготовки:
full  = 1400
train = 1260
val   = 140

после подготовки:
full  = 1398
train = 1258
val   = 140
```

Были исключены только две известные записи с semantic overlap:

```text
tech_804
tech_824
```

SHA-256 подготовленного набора:

```text
full:
21c26864181f0dd455ed4f64a7640fb8d4804f2d4779ec9f851f38375d4c8485

train:
30626160933dfe40bdb4832d5fc36536bbaf1870d6b0f513772f25d8e77519ec

val:
2ca505757f40e6855598021c2d93cf3d1aea27e4e6809b24bbcec0d3b34f9ecd
```

### Почему raw JSONL не публикуются

Сохранённые metadata хорошо описывают:

- количество записей;
- split;
- известные overlaps;
- hashes;
- ограничения аудита.

Но они **не содержат достаточной provenance/licensing информации**, чтобы уверенно утверждать право на публичное перераспространение всех исходных 1400 записей.

Поэтому следующие файлы остаются локальными и исключены из Git:

```text
data/nova_technical_v1_1/nova_technical_v1_1_prepared.jsonl
data/nova_technical_v1_1/splits/train.jsonl
data/nova_technical_v1_1/splits/val.jsonl
```

При этом в repository сохраняются:

```text
data/nova_technical_v1_1/READ_ME.txt
data/nova_technical_v1_1/report.json
```

Это позволяет сохранить экспериментальную историю, hashes и структуру набора без публикации самих записей.

Если provenance и redistribution rights будут отдельно проверены в будущем, dataset можно выпустить отдельным осознанным релизом.

---

## 5. Model checkpoints

Weights не хранятся в Git repository.

### FINAL

Локальный путь:

```text
out/final/best_final.pt
```

SHA-256:

```text
765E3B02A8DA32E30D728BBE6A68F7B5130A5FA6726B0BA5775F22352D83231E
```

Метаданные:

```text
iter_num      = 45000
best_val_loss = 0.515568
parameters    ≈ 124.010M
```

### SFT — 20 steps

Локальный путь:

```text
out/candidate_sft/technical_v1_step20/best_candidate.pt
```

SHA-256:

```text
19B88F804126B085AE70830E2654589EED2899E76DE65804968CE331302EDE90
```

### SFT — 120 steps

Локальный путь:

```text
out/candidate_sft/technical_v1_step120/best_candidate.pt
```

SHA-256:

```text
3C1C3D89C81B16EDF6C982210BEDB33B04512FE25B7DF9BBFF6BFC10DDDD36A8
```

Эти hashes позволяют проверить локальные файлы, если соответствующие weights существуют отдельно.

---

## 6. RAG artifacts

Бинарные FAISS indices и generated index metadata не публикуются.

К ним относятся экспериментальные индексы:

```text
knowledge_index
knowledge_index_v2
technical_index
core_index
question_router
trusted_index
```

Они считаются generated artifacts и могут быть перестроены из соответствующих локальных источников, если эти источники доступны.

---

## 7. Что означает это для воспроизводимости

Обычный:

```text
git clone
```

даёт:

- архитектуру;
- inference code;
- RAG code;
- SFT code;
- tests;
- experiment reports;
- документацию.

Но он **не даёт полностью готовую NOVA_v1**, потому что отсутствуют:

- trained weights;
- полный pretraining corpus;
- raw technical dataset;
- FAISS indices.

Поэтому repository воспроизводит инженерную структуру и историю экспериментов, но не является полностью самодостаточным model release.

---

## 8. Лицензии и права на данные

Лицензия исходного кода и права на распространение datasets — разные вопросы.

Наличие лицензии на код **не означает автоматически**, что те же права распространяются на training data, сторонние тексты, weights или производные datasets.

Raw technical dataset и большие pretraining corpora поэтому не включаются в repository до отдельной проверки provenance и redistribution rights.

Если в корне repository отсутствует файл `LICENSE`, это не следует трактовать как автоматическое разрешение на свободное переиспользование содержимого.

---

## 9. Исторические reports

Файлы в `reports/` сохраняются как экспериментальные артефакты.

Они могут содержать:

- результаты retrieval;
- модельные ответы;
- примеры тестовых prompts;
- исторические абсолютные пути;
- фрагменты диагностических выводов.

Они не переписываются задним числом только ради косметической унификации, если это могло бы изменить смысл исторического результата.

Перед первой публичной публикацией repository был отдельно проверен на типичные сигнатуры API keys, access tokens, private keys и паролей. Найденные совпадения со словом `password` относились к учебным техническим примерам.

---

## 10. Главный принцип архива NOVA_v1

NOVA_v1 сохраняется как честный снимок первой исследовательской итерации.

Если какая-то историческая информация не сохранилась, документация отмечает этот пробел прямо, а не восстанавливает его предположениями.

Дальнейшая разработка выполняется отдельно в `NOVA_v2`.
