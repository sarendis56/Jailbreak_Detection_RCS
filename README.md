# (ACL 2026) Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring

## Setup

### 1. Install LLaVA and Dependencies

```bash
git clone https://github.com/haotian-liu/LLaVA.git
conda create -n llava python=3.10 -y
conda activate llava
cd LLaVA && pip install -e . && cd ..
pip install -r requirements_llava.txt  # additional packages and dependencies required to run our experiments (please do not skip)
```
Do not switch the order of the last two commands above. After running the last, it might tell you about a version mismatch that llava requires an older PyTorch, *it is fine*.

We also provide environment requirement files for different models. Our scripts `kcd.py` and `mcd.py` choose the model from the command line (`--model` / `-m`, or a positional `llava`, `qwen`, or `internvl` argument), then map that choice to a fixed local checkpoint path under `model/`. If no model is specified, they default to `llava`.

**For Qwen2.5-VL:**
```bash
conda create -n qwen25vl python=3.10 -y
conda activate qwen25vl
pip install -r requirements_qwen25vl.txt
```

**For InternVL3-8B:**
```bash
conda create -n internvl3 python=3.10 -y
conda activate internvl3
pip install -r requirements_internvl3.txt
```

### 2. Download Models

The project supports multiple vision-language models such as LLaVA, Qwen, and InternVL. Download all required models:

```bash
python download_models.py
```

This will download the following models to the `model/` directory:
- **LLaVA-v1.6-Vicuna-7B**: Default for most experiments
- **FLAVA**: Facebook's multimodal model for baseline comparisons (`code/baseline_flava.py`)
- **Qwen2.5-VL-3B-Instruct**: Qwen vision-language model
- **Qwen2.5-VL-7B-Instruct**: Larger Qwen model (~13GB, default in our Qwen experiments)
- **InternVL3-8B**: OpenGVLab InternVL3 (~15GB)

Note that you can customize the models to download by editing `MODELS_TO_DOWNLOAD` in `download_models.py` by considering the experiments you want to run and available disk space. If you only need to run the main experiment, you only need to download the specific target model (FLAVA only for the baseline).

The model downloading script will download the model to `./model` locally for faster testing and development. If you wish not to do so, you can skip this step and the model will download when you first run the script.

### 3. Download Datasets

```bash
python download_datasets.py
```
After this, download the rest of the datasets with [this link](https://drive.google.com/file/d/1V09sherPVm6M0E_J_xz3uJ6IBrZ66cRV/view?usp=sharing) (recommended) or manually following the instructions in the terminal.

### 4. Verify Setup

```bash
python verify_setup.py
mkdir results
```

## Running Experiments

Run experiments from the project root directory (*not inside the `code` directory*).

```bash
python code/kcd.py # by default llava is used
python code/mcd.py --model qwen # or -m for controlling which model to run
python code/run_multiple_experiments.py --script kcd --model qwen --runs 5
```

- Scripts `kcd.py`, `mcd.py` are the main scripts of our methods.
- Scripts starting with `hidden_detect_` are our best-effort replication of [HiddenDetect (ACL 2025)](https://github.com/leigest519/hiddendetect) in our scenario, including its proposed layer selection heuristics and detection.
- Use `run_multiple_experiments.py` to run an experiment multiple times and aggregate the results.
- `feature_cache`, `load_datasets`, `profiling_utils`, `feature_extractor*` are helper scripts
- Code in `analysis` can be used to replicate several visualizations such as PCA analysis and visualization of our layer selection heuristics.

## Contact
Please contact Peichun Hua at <peichunhua04@gmail.com> for any question about the code or paper.

## Citation
If you use this code or find our work helpful, please cite:

```bibtex
@misc{hua2025rethinking,
  title        = {Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring},
  author       = {Hua, Peichun and Li, Hao and Shi, Shanghao and Yu, Zhiyuan and Zhang, Ning},
  year         = {2025},
  eprint       = {2512.12069},
  archivePrefix= {arXiv},
  primaryClass = {cs.CR},
  url          = {https://arxiv.org/abs/2512.12069}
}
