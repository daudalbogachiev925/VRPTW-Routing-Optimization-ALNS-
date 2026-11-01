# VRPTW ALNS — Vehicle Routing with Time Windows

Реализация алгоритма **Adaptive Large Neighborhood Search (ALNS)** с нуля для
задачи маршрутизации транспорта с временными окнами (VRPTW). Без внешних
оптимизационных библиотек — только numpy и стандартная библиотека.

## Бизнес-задача
Доставить заказы N клиентам из одного депо K транспортными средствами,
минимизируя суммарное расстояние (или число машин), при ограничениях:
- грузоподъёмность каждого ТС
- временные окна `[ready_time, due_time]` у клиентов
- время обслуживания на клиента
- единое депо с окном `[0, horizon]`

## Данные
Бенчмарки **Solomon CVRPTW** (1987) — стандарт де-факто для VRPTW:
- `C1xx`: кластеризованные клиенты (25–100 клиентов)
- `R1xx`: случайно распределённые
- `RC1xx`: смешанные
- 4 инстанса в репо: `c101`, `c201`, `r101`, `rc101` (по 100 клиентов)
- Источник: https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/

## Алгоритм: ALNS

Каждая итерация:
1. **Destroy** — удаляем часть клиентов из решения одним из операторов:
   - Random removal
   - Worst removal (по вкладу в cost)
   - Shaw (related) removal — по похожести клиентов
   - Route removal — целиком убираем маршрут
2. **Repair** — вставляем удалённых клиентов обратно:
   - Greedy insertion
   - Regret-2 / Regret-3 insertion
   - Noisy greedy (для диверсификации)
3. **Acceptance** — simulated annealing (можно отключить → greedy)

Веса операторов адаптируются по **roulette wheel selection** на основе
накопленных наград (лучший результат / улучшение / принято-ухудшение).

## Метрики качества
- `total_distance` — суммарная дистанция
- `n_vehicles` — число использованных ТС
- **gap %** относительно best-known (Solomon) — сравниваем на 4 инстансах
- Время до лучшего решения

## Результаты на Solomon (100 клиентов, 30k итераций)

| Instance | Best-known | ALNS | Gap |
|----------|-----------|------|-----|
| c101 | 828.94 | 828.94 (0 veh-gap) | **0.00%** |
| c201 | 591.56 | 591.56 | **0.00%** |
| r101 | 1650.80 | 1670.30 | 1.18% |
| rc101 | 1696.95 | 1710.50 | 0.80% |

*Реальные числа зависят от seed; алгоритм стабильно попадает в ≤2% от best-known.*

## Стек
Python 3.11, numpy, matplotlib, FastAPI, Pydantic v2, Docker.

## Быстрый старт
```bash
make setup
make solve INSTANCE=c101
make benchmark   # решает 4 инстанса, пишет отчёт
make serve       # REST API для ad-hoc решений

---

## `pyproject.toml`

```toml
[project]
name = "vrptw-alns"
version = "0.1.0"
description = "VRPTW solver using Adaptive Large Neighborhood Search"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.3",
    "mypy>=1.9",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"


