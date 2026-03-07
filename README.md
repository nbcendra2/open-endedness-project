# QD-Memento: Quality-Diversity Memory Selection for Continual Learning in LLM Agents

## Quick Start

1. Create a new Conda environment:
```bash
conda create -n qd-memento python=3.10 -y
conda activate qd-memento
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=put_your_key_here
```

4. Run evaluator:
```bash
python evaluator.py
```

5. Change experiment settings in:
```bash
config/config.yaml
```
