# VLM Jailbreak Detection

## Setup

### 1. Install LLaVA and Dependencies

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support
pip install -e .

cd ..
pip install -r requirements.txt # For most GPUs except RTX5090
```

We also provide environment requirement files for different models:

**For Qwen2.5-VL:**
```bash
conda create -n qwen25vl python=3.10 -y
conda activate qwen25vl
pip install -r requirements_qwen25vl.txt
```

**For InternVL3-8B:**
If you encounter problems on RTX5090, please open the requirement files and uncomment the newer versions of torch.
```bash
conda create -n internvl3 python=3.10 -y
conda activate internvl3
pip install -r requirements_internvl3.txt
```

**For LLaVA (RTX 5090 compatible):**
If you encounter problems about CUDA compatibility on RTX 5090, try the followinig solution.
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install -r requirements_llava.txt
pip install -e ./LLaVA
```

### 2. Download Models

The project supports multiple vision-language models. For the main experiments, we support LLaVA, Qwen, and InternVL. Download all required models:

```bash
python download_models.py
```

This will download the following models to the `model/` directory:
- **LLaVA-v1.6-Vicuna-7B**: Main multimodal model (~13GB)
- **FLAVA**: Facebook's multimodal model for baseline comparisons
- **CLIP ViT Base**: Baseline vision model
- **CLIP ViT Large**: Vision tower used by LLaVA (~1.7GB)
- **Qwen2.5-VL-3B-Instruct**: Qwen vision-language model (~6GB)
- **Qwen2.5-VL-7B-Instruct**: Larger Qwen model (~13GB)
- **InternVL3-8B**: OpenGVLab InternVL3 (~15GB)

Note that you can customize the models to download by editing `MODELS_TO_DOWNLOAD` in `download_models.py` by considering the experiments you want to run and available disk space. If you only need to run the main experiment, you only need to download the specific target model (FLAVA and CLIP are mostly for baselines).

The model downloading script will download the model to `./model` locally for faster testing and development. If you wish not to do so, you can skip this step and the model will download when you first run the script.

### 3. Vision Tower Configuration For LLaVA

Most people can safely skip this step. LLaVA uses CLIP ViT Large as its vision tower. The model automatically references the vision tower through its configuration:

- Default behavior: LLaVA loads `openai/clip-vit-large-patch14-336` from HuggingFace Hub
- Local override: You can modify the LLaVA config to use the locally downloaded CLIP model

To use the local CLIP model, modify `model/llava-v1.6-vicuna-7b/config.json`:
```json
{
  "mm_vision_tower": "./model/clip-vit-large-patch14-336"
}
```

### 4. Download Datasets

```bash
python download_datasets.py
```

Note that according to the licenses or policies of authors of some datasets, you need to download them manually and put it in the `data` directory. For example, JailbreakV-28k might ask you to fill out a form before you can download. For better accessibility, we have replaced VQAv2 with VizWiz-VQA, which is easier to download. But in some places in the code the function name can still be VQAv2.

### 5. Verify Setup

```bash
python verify_setup.py
mkdir results
```

## Running Experiments

Run experiments from the project root directory (*not inside the `code` directory*):

```bash
mkdir results
python code/main_kcd.py --model llava
```

- Scripts starting with `main_` are those to replicate our main results shown in the main results table.
- Scripts starting with `hidden_detect_` are our best-effort replication of HiddenDetect (ACL 2025)
- Scripts starting with `ablate_` can be used to replicate the ablation study in the appendix
- `feature_extractor_*`, `load_datasets`, `generic_classifier` are helper scripts
- Code in `analysis` can be used to replicate several visualizations such as PCA analysis and ablation study, as well as visualizations of our layer selection heuristics (not put in the paper).
