# Opticonomy PDME Demo

- Run evaluations step by step or in batch

## Clone the Repo
  ```
  git clone https://github.com/kaljuvee/pdme-demo.git
  ```

## Create and Activate the Virtual Environment

- Set up a Python virtual environment and activate it (Windows/VS Code / Bash new Terminal):
  ```
  python -m venv venv
  source venv/Scripts/activate
  ```
  - Set up a Python virtual environment and activate it (Linux):
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
  
- Install dependencies from the `requirements.txt` file:
  ```
  pip install -r requirements.txt
  ```

  ## Run the Streamlit app
  - In VS Code Bash terminal run:
  ```
  streamlit run Home.py
  ```

 ## Run Command Line
  - Run PDME Arena

```
python tests/pdme_arena.py \
    --models_file data/pdme_model_list.csv \
    --eval_type generic \
    --num_prompts 3 \
    --battles_output_file data/generic_battles.csv \
    --elo_output_file data/generic_elo.csv \
    --elo_calibration_model claude-3-opus-20240229 \
    --elo_benchmark_file data/llmarena_elo.csv
  ```
