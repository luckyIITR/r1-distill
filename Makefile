.PHONY: setup setup-dev generate-traces curate train-7b train-32b eval-7b eval-32b serve-7b clean lint test
install:
	uv sync --all-extras

setup:
	. .venv/bin/activate
	
generate-traces:
	python -m src.generation.generate_traces --config configs/generation.yaml

curate:
	python -m src.curation.build_dataset --config configs/curation.yaml

train-7b:
	bash scripts/train_7b.sh

train-32b:
	bash scripts/train_32b.sh

eval-7b:
	bash scripts/eval.sh --model_path checkpoints/qwen2.5-7b-r1-distill

eval-32b:
	bash scripts/eval.sh --model_path checkpoints/qwen2.5-32b-r1-distill

serve-7b:
	bash scripts/serve.sh --model checkpoints/qwen2.5-7b-r1-distill --port 8000

lint:
	ruff check src tests && ruff format --check src tests

format:
	ruff format src tests && ruff check --fix src tests

test:
	pytest -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache .pytest_cache