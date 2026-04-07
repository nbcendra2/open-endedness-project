# Memory Mechanisms for LLM Agents

## Quick Start

1. Create a virtual environment (pick one):

**Conda:**
```bash
conda create -n fade-mem python=3.10 -y
conda activate fade-mem
```

**pip venv:**
```bash
python -m venv fade-mem
source fade-mem/bin/activate        # Mac/Linux
fade-mem\Scripts\activate           # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** TextWorld requires Linux or macOS. Windows users should use WSL (Windows Subsystem for Linux).

3. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=put_your_key_here
GEMINI_API_KEY=put_your_key_here
DEEPSEEK_API_KEY=put_your_key_here
```

4. Run evaluator:

**BabyAI:**
```bash
# Single run (settings from config)
python evaluator.py

# Batch mode — all memory types x all tasks
python evaluator.py --batch
```
Change experiment settings in `config/config.yaml`.

**TextWorld (Treasure Hunter):**
```bash
# Single difficulty (easy | medium | hard | very-hard)
python evaluator_textworld.py --difficulty easy

# Run all difficulties sequentially
python evaluator_textworld.py --difficulty all

# Custom episode count and step limit
python evaluator_textworld.py --difficulty hard --max-steps 200 --num-episodes 20

# Batch mode — all difficulties x memory types
python evaluator_textworld.py --batch

# Batch mode for a specific difficulty
python evaluator_textworld.py --batch --difficulty easy
```
Change experiment settings in `config/config_textworld.yaml`.

**TextWorld Coin Collector (batch):**

> You must `cd` into the experiment folder first before running.

```bash
cd coin_collector_exp
python run_tw_batch.py --config config/coin_collector.yaml --levels L1
```
Change experiment settings in `coin_collector_exp/config/coin_collector.yaml`.
