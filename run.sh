config_path=${CONFIG_PATH:-hparams/your_config_file}  # need modification

python main.py --config_path ${config_path}
python filter.py --config_path ${config_path}
python causal_analysis.py --config_path ${config_path}