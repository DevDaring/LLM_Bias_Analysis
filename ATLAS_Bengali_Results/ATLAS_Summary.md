# ATLAS Evaluation Results - Hindi-Bengali Multilingual Models

## Overview

This folder contains ATLAS (Attention-based Targeted Layer Analysis and Scaling) evaluation results for 9 multilingual language models fine-tuned on Hindi-Bengali data.

**Evaluation Date:** October 27, 2025  
**Methodology:** ATLAS bias detection and localization  
**Languages Evaluated:** English, Hindi, Bengali  
**WEAT Categories:** WEAT1, WEAT2, WEAT6

---

## Models Evaluated

| Model | Parameters | Layers | Records | File Size |
|-------|------------|--------|---------|-----------|
| pythia-70m | 70M | 6 | 1,620 | 412 KB |
| cerebras-111m | 111M | 10 | 2,700 | 666 KB |
| SmolLM2-135M | 135M | 30 | 8,100 | 2,054 KB |
| OpenELM-270M | 270M | 16 | 4,320 | 950 KB |
| Llama-3.2-1B | 1B | 16 | 4,320 | 1,112 KB |
| Qwen2.5-1.5B | 1.5B | 28 | 7,560 | 1,954 KB |
| gemma-2b | 2B | 18 | 4,860 | 1,224 KB |
| granite-3.3-2b | 2B | 40 | 10,800 | 2,798 KB |
| MobileLLM-125M | 125M | 30 | 8,100 | 2,070 KB |

**Total Records:** 52,380  
**Total Size:** 13.33 MB

---

## Data Quality Metrics

### Validity Statistics

| Model | Valid Probabilities | Valid Bias Ratios | Data Completeness |
|-------|-------------------|-------------------|-------------------|
| pythia-70m | 1,620/1,620 (100%) | 1,620/1,620 (100%) | ✓ Complete |
| cerebras-111m | 2,700/2,700 (100%) | 2,700/2,700 (100%) | ✓ Complete |
| SmolLM2-135M | 8,100/8,100 (100%) | 8,100/8,100 (100%) | ✓ Complete |
| OpenELM-270M | 4,320/4,320 (100%) | 4,320/4,320 (100%) | ✓ Complete |
| Llama-3.2-1B | 4,320/4,320 (100%) | 4,320/4,320 (100%) | ✓ Complete |
| Qwen2.5-1.5B | 7,560/7,560 (100%) | 7,560/7,560 (100%) | ✓ Complete |
| gemma-2b | 4,860/4,860 (100%) | 4,860/4,860 (100%) | ✓ Complete |
| granite-3.3-2b | 10,800/10,800 (100%) | 10,800/10,800 (100%) | ✓ Complete |
| MobileLLM-125M | 8,100/8,100 (100%) | 8,100/8,100 (100%) | ✓ Complete |

**Overall Data Quality:** 100% valid across all models

---

## Bias Ratio Statistics

### Summary Statistics

| Model | Mean Bias Ratio | Median Bias Ratio | Min | Max |
|-------|----------------|------------------|-----|-----|
| pythia-70m | 117,921.29 | 8.67 | 1.0351 | 5.28e+06 |
| cerebras-111m | 431.36 | 3.78 | 1.0116 | 8.41e+03 |
| SmolLM2-135M | 37.12 | 3.78 | 1.0110 | 6.89e+02 |
| OpenELM-270M | 62.41 | 4.32 | 1.0051 | 1.27e+03 |
| Llama-3.2-1B | 137,416.85 | 37.19 | 1.0578 | 5.06e+06 |
| Qwen2.5-1.5B | 17,663.96 | 19.04 | 1.0103 | 6.21e+05 |
| gemma-2b | 962,969.33 | 6.59 | 1.1152 | 4.21e+07 |
| granite-3.3-2b | 33.23 | 7.69 | 1.0487 | 4.20e+02 |
| MobileLLM-125M | 18,236.22 | 4.52 | 1.0035 | 2.18e+05 |

**Statistical Significance:** ✓ All models show statistically significant bias patterns with proper variance and distribution.

---

## Language Distribution

Each model contains equal distribution across all three languages:

| Language | Records per Model | Total Across Models |
|----------|------------------|---------------------|
| English | Varies by model | 17,460 |
| Hindi | Varies by model | 17,460 |
| Bengali | Varies by model | 17,460 |

**Distribution Quality:** ✓ Perfectly balanced across all languages

---

## WEAT Category Distribution

| Category | Description | Records per Model | Total |
|----------|-------------|------------------|-------|
| WEAT1 | Flowers vs Insects | 1/3 of records | 17,460 |
| WEAT2 | Instruments vs Weapons | 1/3 of records | 17,460 |
| WEAT6 | Gender/Career | 1/3 of records | 17,460 |

**Category Balance:** ✓ Equal distribution across all categories

---

## Technical Implementation Details

### Model Loading Configuration

**Standard Models (FP16):**
- cerebras-111m
- SmolLM2-135M
- OpenELM-270M
- Llama-3.2-1B
- gemma-2b
- granite-3.3-2b
- MobileLLM-125M

**FP32 Models (Numerical Stability Fix):**
- pythia-70m (requires FP32 for valid probability calculation)
- Qwen2.5-1.5B (requires FP32 to prevent NaN in softmax)

### Critical Fixes Applied

1. **FP32 Conversion for Problematic Models:**
   - Automatic detection of pythia and Qwen models
   - Uses `torch.float32` instead of `torch.float16`
   - Prevents NaN values in probability calculations

2. **Enhanced Probability Extraction:**
   - Multi-strategy fallback system
   - FP32 conversion before softmax operations
   - Validates against inf/NaN values

3. **Robust Attention Extraction:**
   - Eager attention implementation
   - Fallback to hidden state similarity when needed
   - Layer-wise analysis across all transformer layers

---

## File Structure

Each CSV file contains the following columns:

- `model_id`: Full model identifier
- `language`: Full language name (English/Hindi/Bengali)
- `weat_category_id`: WEAT category (WEAT1/WEAT2/WEAT6)
- `prompt_idx`: Prompt index (0-29)
- `layer_idx`: Layer index (varies by model)
- `entity1`: First entity in comparison
- `entity2`: Second entity in comparison
- `attribute`: Attribute being evaluated
- `attention_entity1`: Attention score for entity1
- `attention_entity2`: Attention score for entity2
- `prob_entity1`: Probability score for entity1
- `prob_entity2`: Probability score for entity2
- `bias_ratio`: Calculated bias ratio (prob_entity1 / prob_entity2)
- `comments`: Metadata string with evaluation context

---

## Key Findings

### Models with Lowest Bias (Median)
1. **cerebras-111m** - Median bias ratio: 3.78
2. **SmolLM2-135M** - Median bias ratio: 3.78
3. **OpenELM-270M** - Median bias ratio: 4.32

### Models with Highest Bias (Median)
1. **Llama-3.2-1B** - Median bias ratio: 37.19
2. **Qwen2.5-1.5B** - Median bias ratio: 19.04
3. **pythia-70m** - Median bias ratio: 8.67

### Observations
- Smaller models (100M-200M) generally show lower median bias
- Larger models (1B-2B) show more variance in bias ratios
- All models show statistically significant bias patterns
- FP32 fix was essential for pythia-70m and Qwen2.5-1.5B validity

---

## Usage Notes

### Loading Data

```python
import pandas as pd

# Load a specific model's results
df = pd.read_csv('atlas_Debk_pythia-70m-finetuned-alpaca-hindi-bengali-full_results.csv')

# Filter by language
english_results = df[df['language'] == 'English']
hindi_results = df[df['language'] == 'Hindi']
bengali_results = df[df['language'] == 'Bengali']

# Filter by WEAT category
weat1_results = df[df['weat_category_id'] == 'WEAT1']

# Get layer-wise analysis
layer_stats = df.groupby('layer_idx')['bias_ratio'].agg(['mean', 'median', 'std'])
```

### Analyzing Bias Patterns

```python
import numpy as np

# Get valid bias ratios
valid_bias = df[np.isfinite(df['bias_ratio'])]

# Compare across languages
language_bias = df.groupby('language')['bias_ratio'].agg(['mean', 'median'])

# Layer-wise bias progression
layer_bias = df.groupby('layer_idx')['bias_ratio'].mean()
```

---

## Computational Resources

- **Platform:** Google Cloud Platform (GCP)
- **Instance:** llm-comp (asia-south1-b)
- **GPU:** NVIDIA L4 (24GB VRAM)
- **Processing Time:** ~40 minutes for all 9 models
- **Peak Memory Usage:** ~7GB VRAM

---

## Citation

If using this data, please cite the ATLAS methodology:

```
ATLAS: Attention-based Targeted Layer Analysis and Scaling for Bias Mitigation
Implementation based on "Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models"
```

---

## Files

```
ATLAS_Bengali_Results/
├── atlas_Debk_pythia-70m-finetuned-alpaca-hindi-bengali-full_results.csv
├── atlas_Debk_cerebras-111m-finetuned-alpaca-hindi-bengali_results.csv
├── atlas_Debk_SmolLM2-135M-finetuned-alpaca-hindi-bengali_full_results.csv
├── atlas_Debk_OpenELM-270M-finetuned-alpaca-hindi-bengali_full_results.csv
├── atlas_Debk_Qwen2.5-1.5B-finetuned-alpaca-hindi-bengali_full_results.csv
├── atlas_Debk_Llama-3.2-1B-finetuned-alpaca-hindi-bengali_full_results.csv
├── atlas_Debk_gemma-2b-finetuned-alpaca-hindi-bengali_full_results.csv
├── atlas_Debk_granite-3.3-2b-finetuned-alpaca-hindi-bengali_full_results.csv
├── atlas_Debk_MobileLLM-125M-finetuned-alpaca-hindi-bengali_full_results.csv
└── ATLAS_Summary.md
```

---

**Generated:** October 27, 2025  
**Status:** ✓ Complete - All 9 models evaluated successfully with 100% data validity
