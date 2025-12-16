# Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring

## Setup

### 1. Install LLaVA and Dependencies

```bash
git clone https://github.com/haotian-liu/LLaVA.git
conda create -n llava python=3.10 -y
conda activate llava
pip install -e ./LLaVA
pip install -r requirements_llava.txt
```

We also provide environment requirement files for different models:

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

Note that according to the licenses or policies of authors of some datasets, you need to download them manually and put it in the `data` directory. For example, JailbreakV-28k might ask you to fill out a form before you can download.

### 4. Verify Setup

```bash
python verify_setup.py
mkdir results
```

## Running Experiments

Run experiments from the project root directory (*not inside the `code` directory*).

```bash
python code/kcd.py
python code/run_multiple_experiments.py --script kcd --model qwen --runs 5
```

- Scripts `kcd.py`, `mcd.py` are the main scripts of our methods.
- Scripts starting with `hidden_detect_` are our best-effort replication of HiddenDetect (ACL 2025), including the layer selection heuristics and detection.
- Use `run_multiple_experiments.py` to run an experiment multiple times and aggregate the results.
- `feature_cache`, `load_datasets`, `profiling_utils`, `feature_extractor*` are helper scripts
- Code in `analysis` can be used to replicate several visualizations such as PCA analysis and visualization of our layer selection heuristics.

## Contact
Please contact Peichun Hua at <peichunhua04@gmail.com> for any question about the code or paper instead of the WashU email in the paper (because I have left WashU and do not have access to the mailbox anymore).

## Citation

@misc{hua2025rethinkingjailbreakdetectionlarge,
      title={Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring}, 
      author={Peichun Hua and Hao Li and Shanghao Shi and Zhiyuan Yu and Ning Zhang},
      year={2025},
      eprint={2512.12069},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2512.12069}, 
}
