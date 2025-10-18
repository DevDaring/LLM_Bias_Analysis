# ATLAS Results Directory - Index

**Complete ATLAS Evaluation for 12 Large Language Models**

---

## 📂 Directory Contents

### 📊 Result Files (12 CSV files)

| File | Size | Rows | Model Type | Language(s) | Status |
|------|------|------|------------|-------------|--------|
| `atlas_apple_OpenELM-270M_results.csv` | 261 KB | 480 | Base | EN | ✅ |
| `atlas_facebook_MobileLLM-125M_results.csv` | 521 KB | 900 | Base | EN | ✅ |
| `atlas_cerebras_Cerebras-GPT-111M_results.csv` | 167 KB | 300 | Base | EN | ✅ |
| `atlas_EleutherAI_pythia-70m_results.csv` | 34 KB | 180 | Base | EN | ✅ Fixed |
| `atlas_meta-llama_Llama-3.2-1B_results.csv` | 260 KB | 480 | Base | EN | ✅ |
| `atlas_Qwen_Qwen2.5-1.5B_results.csv` | 156 KB | 840 | Base | EN | ✅ Fixed |
| `atlas_DebK_OpenELM-270M-finetuned-alpaca-hindi_full_results.csv` | 642 KB | 960 | Finetuned | EN+HI | ✅ |
| `atlas_DebK_MobileLLM-125M-finetuned-alpaca-hindi_results.csv` | 1.2 MB | 1800 | Finetuned | EN+HI | ✅ |
| `atlas_DebK_cerebras-gpt-111m-finetuned-alpaca-hindi_results.csv` | 404 KB | 600 | Finetuned | EN+HI | ✅ |
| `atlas_DebK_pythia-70m-finetuned-alpaca-hindi_results.csv` | 82 KB | 360 | Finetuned | EN+HI | ✅ Fixed |
| `atlas_DebK_Llama-3.2-1B-finetuned-alpaca-hindi_full_results.csv` | 648 KB | 960 | Finetuned | EN+HI | ✅ |
| `atlas_DebK_Qwen2.5-1.5B-finetuned-alpaca-hindi_full_results.csv` | 394 KB | 1680 | Finetuned | EN+HI | ✅ Fixed |

**Total**: 5.6 MB, 9,540 rows (4,500 unique data points across layers)

---

### 📖 Documentation Files

#### 1. **ATLAS_COMPREHENSIVE_RESULTS.md** (21 KB)
   - **Purpose**: Complete technical documentation
   - **Contents**:
     - Full methodology and mathematical foundations
     - Model specifications and architecture details
     - Input data (WEAT categories, word lists)
     - Evaluation process and pipeline
     - Complete results overview
     - Technical implementation details
     - Challenges encountered and solutions
     - Statistical analysis framework
     - Reproducibility guide
   - **Audience**: Researchers, technical implementers
   - **Use when**: Need detailed understanding of methodology

#### 2. **README.md** (8 KB)
   - **Purpose**: Quick reference and getting started guide
   - **Contents**:
     - Overall statistics summary
     - File listing with descriptions
     - Key findings and bias rankings
     - Data quality metrics
     - Quick code examples (Python)
     - Column reference
     - Visualization examples
     - Validation checklist
   - **Audience**: Data analysts, quick users
   - **Use when**: Need to quickly understand and use the data

#### 3. **EXECUTION_SUMMARY.md** (12 KB)
   - **Purpose**: Project execution timeline and status
   - **Contents**:
     - Complete execution timeline (Phase 1-4)
     - Initial results (8/12 successful)
     - Root cause analysis (FP16 overflow)
     - Solution implementation (FP32 fix)
     - Fixed results (12/12 successful)
     - Technical achievements
     - Performance metrics
     - Validation results
     - Key learnings
   - **Audience**: Project managers, stakeholders
   - **Use when**: Need project status and history

#### 4. **INDEX.md** (This file)
   - **Purpose**: Directory navigation and file reference
   - **Contents**: File listings, purposes, navigation guide

---

## 🎯 Quick Start Guide

### For Researchers
1. Start with **ATLAS_COMPREHENSIVE_RESULTS.md** for methodology
2. Review **README.md** for data schema
3. Load CSV files for analysis

### For Data Analysts
1. Read **README.md** for quick reference
2. Use Python examples to load data
3. Refer to column reference for field meanings

### For Project Stakeholders
1. Check **EXECUTION_SUMMARY.md** for project status
2. Review **README.md** for key findings
3. See validation results

---

## 📈 Data Overview

### Summary Statistics
- **Total Models**: 12 (6 base + 6 finetuned)
- **Total Data Points**: 4,500
- **Total Rows (with layers)**: 9,540
- **Languages**: 2 (English + Hindi)
- **WEAT Categories**: 3 (WEAT1, WEAT6, WEAT7)
- **Data Completeness**: 100%

### Model Categories

**Base Models (6)**:
- Size range: 70M - 1.5B parameters
- Layers: 6 - 28
- All evaluated on English

**Finetuned Models (6)**:
- Same base architectures
- Finetuned on Alpaca-Hindi dataset
- Evaluated on both English and Hindi

---

## 🔍 What Each File Contains

### CSV Files Structure

All CSV files share the same schema (14 columns):

```
model_id, language, weat_category_id, prompt_idx, layer_idx,
entity1, entity2, attribute,
attention_entity1, attention_entity2,
prob_entity1, prob_entity2, bias_ratio,
comments
```

**Key Metrics**:
- `bias_ratio = prob_entity1 / prob_entity2`
- Values > 1: Bias toward entity1
- Values < 1: Bias toward entity2
- Values ≈ 1: Neutral

---

## 🎓 Understanding Bias Ratios

### Interpretation

| Bias Ratio | Interpretation | Example Models |
|------------|----------------|----------------|
| > 100 | Very strong bias toward entity1 | Llama-3.2-1B (151.5) |
| 10-100 | Strong bias | Llama-finetuned (98.3), Pythia (11.87) |
| 2-10 | Moderate bias | MobileLLM (5.72), OpenELM (3.48) |
| 1-2 | Mild bias | Cerebras (2.11) |
| 0.1-1 | Inverse bias (favors entity2) | Qwen (0.016) |

---

## 🛠️ Common Tasks

### Load and Analyze Data

```python
import pandas as pd

# Load a model's results
df = pd.read_csv('atlas_apple_OpenELM-270M_results.csv')

# Basic analysis
print(f"Total layers: {df['layer_idx'].nunique()}")
print(f"Average bias: {df['bias_ratio'].mean():.4f}")

# Filter by WEAT category
weat1 = df[df['weat_category_id'] == 'WEAT1']
print(f"WEAT1 bias: {weat1['bias_ratio'].mean():.4f}")

# Layer-wise analysis
layer_avg = df.groupby('layer_idx')['bias_ratio'].mean()
print(layer_avg)
```

### Compare Models

```python
import pandas as pd
import glob

# Load all base models
files = glob.glob('atlas_*_results.csv')
base_files = [f for f in files if 'finetuned' not in f]

results = {}
for file in base_files:
    df = pd.read_csv(file)
    model_name = file.replace('atlas_', '').replace('_results.csv', '')
    results[model_name] = df['bias_ratio'].mean()

# Print ranking
for model, bias in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model}: {bias:.4f}")
```

### Visualize Bias Across Layers

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('atlas_apple_OpenELM-270M_results.csv')

# Plot bias by layer
layer_bias = df.groupby('layer_idx')['bias_ratio'].mean()

plt.figure(figsize=(12, 6))
plt.plot(layer_bias.index, layer_bias.values, marker='o')
plt.xlabel('Layer')
plt.ylabel('Average Bias Ratio')
plt.title('Bias Evolution Across Layers - OpenELM-270M')
plt.grid(True)
plt.savefig('bias_across_layers.png')
plt.show()
```

---

## ✅ Validation Status

### Data Quality Checks

All 12 CSV files passed validation:

```
✅ No missing values in critical columns
✅ All bias_ratio values are valid numbers (not NaN)
✅ All probability values > 0
✅ All attention scores in range [0, 1]
✅ Consistent schema across all files
✅ Expected row counts for each model
```

### Fixed Models

4 models required fixes (FP32 solution):
- ✅ EleutherAI/pythia-70m
- ✅ DebK/pythia-70m-finetuned-alpaca-hindi
- ✅ Qwen/Qwen2.5-1.5B
- ✅ DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full

**Fix Applied**: Changed softmax computation from FP16 to FP32 to prevent numerical overflow.

---

## 📞 Need Help?

### For Questions About:

**Methodology**: → Read `ATLAS_COMPREHENSIVE_RESULTS.md`
- Section: "Methodology"
- Section: "Technical Implementation"

**Data Usage**: → Read `README.md`
- Section: "Quick Analysis"
- Section: "Column Reference"

**Project Status**: → Read `EXECUTION_SUMMARY.md`
- Section: "Final Status"
- Section: "Validation Results"

**Specific Models**: → Check individual CSV file
- Row count matches expected?
- Check bias_ratio column for values
- Verify no NaN in critical columns

---

## 📊 File Size Summary

```
Documentation:
  ATLAS_COMPREHENSIVE_RESULTS.md:  21 KB
  README.md:                        8 KB
  EXECUTION_SUMMARY.md:            12 KB
  INDEX.md:                         6 KB
  ─────────────────────────────────────
  Total Documentation:             47 KB

CSV Results:
  Base models (6 files):          1.4 MB
  Finetuned models (6 files):     4.2 MB
  ─────────────────────────────────────
  Total Results:                  5.6 MB

GRAND TOTAL:                     ~5.7 MB
```

---

## 🎯 Success Metrics

- ✅ 12/12 models evaluated (100%)
- ✅ 4,500 data points collected
- ✅ 100% data completeness
- ✅ 0 validation failures
- ✅ Complete documentation
- ✅ Reproducible pipeline

---

## 📅 Timeline

- **Initial Run**: October 18, 2025 (05:40-06:02) - 8/12 successful
- **Analysis**: October 18, 2025 (06:02-06:15) - Identified FP16 issue
- **Fixed Run**: October 18, 2025 (06:30-06:33) - 4/4 successful
- **Documentation**: October 18, 2025 (12:00-12:15) - Complete
- **Status**: ✅ **COMPLETE**

---

## 🏆 Final Status

```
╔════════════════════════════════════════════════╗
║                                                ║
║   ✅ ATLAS EVALUATION COMPLETE                ║
║                                                ║
║   Models:     12/12 (100%)                    ║
║   Data:       4,500 points                    ║
║   Quality:    100% complete                   ║
║   Docs:       4 comprehensive files           ║
║                                                ║
║   STATUS: READY FOR ANALYSIS                  ║
║                                                ║
╚════════════════════════════════════════════════╝
```

---

**Last Updated**: October 18, 2025  
**Directory**: `./ATLAS_Results/`  
**Status**: ✅ Complete and Validated
