# -*- coding: utf-8 -*-
"""
Parallel Enhanced CEAT (Contextual Embedding Association Test) for Layer-wise Bias Analysis
Fixed version with proper batching and multiprocessing
Based on: https://arxiv.org/pdf/2006.03955
"""

import os
import gc
import sys
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from huggingface_hub import login
import warnings
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import psutil

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')

# Set HF token
HF_TOKEN = "secret"


# Model configurations with size annotations
BASE_MODELS = [
    "apple/OpenELM-270M",           # 270M
    "facebook/MobileLLM-125M",      # 125M
    "cerebras/Cerebras-GPT-111M",   # 111M
    "EleutherAI/pythia-70m",        # 70M
    "meta-llama/Llama-3.2-1B",      # 1B
    "Qwen/Qwen2.5-1.5B",            # 1.5B
    "HuggingFaceTB/SmolLM2-135M"    # 135M
]

FINETUNED_MODELS = [
    "DebK/pythia-70m-finetuned-alpaca-hindi",              # 70M
    "DebK/cerebras-gpt-111m-finetuned-alpaca-hindi",       # 111M
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi",          # 125M
    "DebK/OpenELM-270M-finetuned-alpaca-hindi_full",       # 270M
    "DebK/Llama-3.2-1B-finetuned-alpaca-hindi_full",       # 1B
    "DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full",       # 1.5B
    "DebK/SmolLM2-135M-finetuned-alpaca-hindi"             # 135M
]

# Model size mapping (in millions of parameters)
MODEL_SIZES = {
    "apple/OpenELM-270M": 270,
    "facebook/MobileLLM-125M": 125,
    "cerebras/Cerebras-GPT-111M": 111,
    "EleutherAI/pythia-70m": 70,
    "meta-llama/Llama-3.2-1B": 1000,
    "Qwen/Qwen2.5-1.5B": 1500,
    "HuggingFaceTB/SmolLM2-135M": 135,
    "DebK/pythia-70m-finetuned-alpaca-hindi": 70,
    "DebK/cerebras-gpt-111m-finetuned-alpaca-hindi": 111,
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi": 125,
    "DebK/OpenELM-270M-finetuned-alpaca-hindi_full": 270,
    "DebK/Llama-3.2-1B-finetuned-alpaca-hindi_full": 1000,
    "DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full": 1500,
    "DebK/SmolLM2-135M-finetuned-alpaca-hindi": 135
}

# Special tokenizer mappings
TOKENIZER_MAPPING = {
    "apple/OpenELM-270M": "meta-llama/Llama-2-7b-hf",
    "DebK/OpenELM-270M-finetuned-alpaca-hindi_full": "meta-llama/Llama-2-7b-hf",
    "facebook/MobileLLM-125M": "meta-llama/Llama-2-7b-hf",
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi": "meta-llama/Llama-2-7b-hf"
}

# Models requiring special tokenizer loading configuration
SPECIAL_TOKENIZER_CONFIG = {
    "facebook/MobileLLM-125M": {"use_fast": False},
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi": {"use_fast": False},
    "cerebras/Cerebras-GPT-111M": {"use_fast": False},
    "DebK/cerebras-gpt-111m-finetuned-alpaca-hindi": {"use_fast": False}
}

# Define sentence templates globally to avoid pickling issues
SENTENCE_TEMPLATES = {
    'en': {
        'WEAT1': [
            "Researchers discovered that {} exhibits unique characteristics.",
            "The study of {} reveals important biological patterns.",
            "Looking at the {}, one feels a sense of tranquility.",
            "The {} evokes strong emotional responses in observers.",
            "The {} moves gracefully through its environment.",
            "Children often chase after {} in the garden.",
            "Unlike other species, {} demonstrates remarkable adaptation.",
            "The {} stands out among its peers in the natural world."
        ],
        'WEAT2': [
            "The musician skillfully played the {} during the concert.",
            "The soldier carefully maintained his {} before battle.",
            "In many cultures, the {} symbolizes tradition and heritage.",
            "The {} holds deep cultural significance across generations.",
            "Students learn to appreciate the {} through practice.",
            "Mastering the {} requires years of dedicated training.",
            "The community gathered to witness the {} demonstration.",
            "People from all walks of life admire the {}."
        ],
        'WEAT6': [
            "{} demonstrated exceptional leadership in the boardroom.",
            "{} balanced work responsibilities with personal commitments.",
            "{} participated actively in community events.",
            "{} formed meaningful relationships with colleagues.",
            "{} achieved recognition for outstanding contributions.",
            "{} overcame challenges through determination and skill.",
            "{} valued time spent with family and friends.",
            "{} pursued hobbies that brought personal fulfillment."
        ]
    },
    'hi': {
        'WEAT1': [
            "वैज्ञानिकों ने पाया कि {} अद्वितीय गुण प्रदर्शित करता है।",
            "{} का अध्ययन महत्वपूर्ण जैविक पैटर्न प्रकट करता है।",
            "{} को देखकर शांति का अनुभव होता है।",
            "{} दर्शकों में मजबूत भावनात्मक प्रतिक्रिया उत्पन्न करता है।",
            "{} अपने वातावरण में सुंदरता से घूमता है।",
            "बच्चे बगीचे में {} के पीछे भागते हैं।",
            "अन्य प्रजातियों के विपरीत, {} उल्लेखनीय अनुकूलन प्रदर्शित करता है।",
            "{} प्राकृतिक दुनिया में अपने साथियों में अलग है।"
        ],
        'WEAT2': [
            "संगीतकार ने संगीत समारोह में कुशलता से {} बजाया।",
            "सैनिक ने युद्ध से पहले अपने {} की देखभाल की।",
            "कई संस्कृतियों में, {} परंपरा और विरासत का प्रतीक है।",
            "{} पीढ़ियों में गहरा सांस्कृतिक महत्व रखता है।",
            "छात्र अभ्यास के माध्यम से {} की सराहना करना सीखते हैं।",
            "{} में महारत हासिल करने के लिए वर्षों के समर्पित प्रशिक्षण की आवश्यकता है।",
            "समुदाय {} प्रदर्शन देखने के लिए एकत्र हुआ।",
            "सभी वर्गों के लोग {} की प्रशंसा करते हैं।"
        ],
        'WEAT6': [
            "{} ने बोर्डरूम में असाधारण नेतृत्व प्रदर्शित किया।",
            "{} ने कार्य जिम्मेदारियों को व्यक्तिगत प्रतिबद्धताओं के साथ संतुलित किया।",
            "{} ने सामुदायिक कार्यक्रमों में सक्रिय रूप से भाग लिया।",
            "{} ने सहकर्मियों के साथ सार्थक संबंध बनाए।",
            "{} ने उत्कृष्ट योगदान के लिए मान्यता प्राप्त की।",
            "{} ने दृढ़ संकल्प और कौशल के माध्यम से चुनौतियों को पार किया।",
            "{} ने परिवार और दोस्तों के साथ बिताए समय को महत्व दिया।",
            "{} ने शौक का पीछा किया जो व्यक्तिगत संतुष्टि लाते थे।"
        ]
    }
}

def log_debug(message, level=1):
    """Print debug messages with indentation based on level"""
    indent = "  " * (level - 1)
    print(f"{indent}DEBUG: {message}")

def get_gpu_memory_info():
    """Get GPU memory information"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_free': gpu.memoryFree,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_util': gpu.memoryUtil * 100
                })
            return gpu_info
    except:
        pass
    return []

def create_smart_batches(models):
    """
    Create smart batches ensuring:
    1. Maximum 3 models per batch
    2. No two large models (≥1B params) in the same batch
    3. Optimal GPU utilization
    """
    
    # Categorize models by size
    small_models = []  # < 300M params
    medium_models = [] # 300M - 999M params  
    large_models = []  # ≥ 1B params
    
    for model in models:
        size = MODEL_SIZES.get(model, 200)
        if size >= 1000:
            large_models.append((model, size))
        elif size >= 300:
            medium_models.append((model, size))
        else:
            small_models.append((model, size))
    
    # Sort each category by size
    small_models.sort(key=lambda x: x[1])
    medium_models.sort(key=lambda x: x[1])
    large_models.sort(key=lambda x: x[1])
    
    batches = []
    
    # Strategy 1: Each large model gets its own batch with up to 2 small models
    for large_model, large_size in large_models:
        batch = [large_model]
        
        # Add up to 2 small models if available
        small_count = min(2, len(small_models))
        for _ in range(small_count):
            if small_models:
                batch.append(small_models.pop(0)[0])
        
        # If no small models, add 1 medium model if total size is reasonable
        if len(batch) == 1 and medium_models:
            # Only add medium if total would be < 2000M params
            if medium_models[0][1] + large_size < 2000:
                batch.append(medium_models.pop(0)[0])
        
        batches.append(batch)
    
    # Strategy 2: Group remaining medium models (max 3 per batch)
    while medium_models:
        batch = []
        batch_size = 0
        
        for _ in range(3):
            if medium_models and batch_size + medium_models[0][1] < 1500:
                model, size = medium_models.pop(0)
                batch.append(model)
                batch_size += size
            else:
                break
        
        if batch:
            batches.append(batch)
    
    # Strategy 3: Group remaining small models (max 3 per batch)
    while small_models:
        batch = []
        for _ in range(3):
            if small_models:
                batch.append(small_models.pop(0)[0])
        
        if batch:
            batches.append(batch)
    
    return batches

class WEATHubLoader:
    """Loads the WEATHub dataset and provides word lists."""
    def __init__(self, dataset_id: str, cache_dir: str = None):
        print(f"Loading WEATHub dataset from '{dataset_id}'...")
        try:
            self.dataset = load_dataset(dataset_id, cache_dir=cache_dir)
            print("WEATHub dataset loaded successfully.")
            self.split_mapping = {
                'WEAT1': 'original_weat', 'WEAT2': 'original_weat', 'WEAT6': 'original_weat', 
                'WEAT7': 'original_weat', 'WEAT8': 'original_weat'
            }
        except Exception as e:
            print(f"ERROR: Failed to load WEATHub dataset. Exception: {e}")
            self.dataset = None

    def get_word_lists(self, language_code: str, weat_category_id: str):
        """Retrieves target and attribute word lists."""
        if not self.dataset: 
            return None
        split_name = self.split_mapping.get(weat_category_id)
        if not split_name:
            print(f"Warning: Category '{weat_category_id}' not found.")
            return None
        try:
            filtered = self.dataset[split_name].filter(
                lambda x: x['language'] == language_code and x['weat'] == weat_category_id
            )
            if len(filtered) > 0:
                return {
                    'targ1': filtered[0]['targ1.examples'], 
                    'targ2': filtered[0]['targ2.examples'], 
                    'attr1': filtered[0]['attr1.examples'], 
                    'attr2': filtered[0]['attr2.examples']
                }
            else:
                print(f"Warning: No data for language '{language_code}' and category '{weat_category_id}'.")
                return None
        except Exception as e:
            print(f"Error filtering data for '{weat_category_id}' in language '{language_code}': {e}")
            return None

class EnhancedCEATEvaluator:
    """Enhanced CEAT evaluation with proper random-effects meta-analysis"""
    
    def __init__(self, cache_dir="./hf_cache", device_id=None):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.device_id = device_id
        self.sentence_templates = SENTENCE_TEMPLATES  # Use global templates
        self.model = None
        self.tokenizer = None
        self.current_model_id = None
        
    def clear_model_cache(self):
        """Clears the current model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_id = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def load_model(self, model_id):
        """Load model with appropriate configuration"""
        self.clear_model_cache()
        
        tokenizer_id = TOKENIZER_MAPPING.get(model_id, model_id)
        
        print(f"[Device {self.device_id}] Loading model: {model_id}")
        
        trust_remote = any(model_name in model_id for model_name in ["OpenELM", "MobileLLM"])
        tokenizer_config = SPECIAL_TOKENIZER_CONFIG.get(model_id, {"use_fast": True})
        
        # Set HF token for this process
        login(HF_TOKEN, add_to_git_credential=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id, 
                cache_dir=self.cache_dir,
                token=HF_TOKEN,
                trust_remote_code=trust_remote,
                **tokenizer_config
            )
            
        except Exception as e:
            try:
                opposite_fast = not tokenizer_config.get("use_fast", True)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    cache_dir=self.cache_dir,
                    token=HF_TOKEN,
                    trust_remote_code=trust_remote,
                    use_fast=opposite_fast
                )
                
            except Exception as e2:
                try:
                    fallback_tokenizer = "meta-llama/Llama-2-7b-hf"
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        fallback_tokenizer,
                        cache_dir=self.cache_dir,
                        token=HF_TOKEN,
                        trust_remote_code=False
                    )
                    
                except Exception as e3:
                    print(f"[Device {self.device_id}] Error: All tokenizer loading attempts failed for {model_id}")
                    return False
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            # Specify device map based on device_id
            if self.device_id is not None and torch.cuda.device_count() > 1:
                device_map = f"cuda:{self.device_id}"
            else:
                device_map = "auto"
            
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": device_map,
                "cache_dir": self.cache_dir,
                "token": HF_TOKEN,
                "trust_remote_code": trust_remote,
                "low_cpu_mem_usage": True
            }
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            self.current_model_id = model_id
            print(f"[Device {self.device_id}] Model {model_id} loaded successfully")
            
        except Exception as e:
            print(f"[Device {self.device_id}] Error loading model {model_id}: {str(e)}")
            return False
        
        return True
    
    def find_word_token_positions_improved(self, sentence: str, target_word: str):
        """Improved word tokenization handling with better subword support"""
        tokens = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
        token_ids = tokens['input_ids'][0]
        offset_mapping = tokens['offset_mapping'][0]
        
        target_positions = []
        sentence_lower = sentence.lower()
        target_lower = target_word.lower()
        
        start_idx = 0
        while True:
            word_start = sentence_lower.find(target_lower, start_idx)
            if word_start == -1:
                break
                
            word_end = word_start + len(target_lower)
            
            overlapping_tokens = []
            for i, (start_char, end_char) in enumerate(offset_mapping):
                if start_char == 0 and end_char == 0:
                    continue
                if not (end_char <= word_start or start_char >= word_end):
                    overlapping_tokens.append(i)
            
            target_positions.extend(overlapping_tokens)
            start_idx = word_end
        
        return list(set(target_positions))
    
    @torch.no_grad()
    def get_contextual_word_embedding(self, sentence: str, target_word: str, layer_idx: int):
        """Get contextual embedding of a target word from a specific layer"""
        try:
            token_positions = self.find_word_token_positions_improved(sentence, target_word)
            
            if not token_positions:
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    return np.zeros(self.model.config.hidden_size)
                else:
                    return np.zeros(768)
            
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_idx][0]
            
            word_embeddings = []
            for pos in token_positions:
                if pos < hidden_state.shape[0]:
                    word_embeddings.append(hidden_state[pos].cpu().numpy())
            
            if not word_embeddings:
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    return np.zeros(self.model.config.hidden_size)
                else:
                    return np.zeros(768)
            
            word_embedding = np.mean(word_embeddings, axis=0)
            
            return word_embedding.astype(np.float32)
            
        except Exception as e:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                return np.zeros(self.model.config.hidden_size)
            else:
                return np.zeros(768)
    
    def _get_template_embeddings(self, words, template, layer_idx):
        """Get embeddings for words in a specific template"""
        embeddings = []
        for word in words:
            sentence = template.format(word)
            emb = self.get_contextual_word_embedding(sentence, word, layer_idx)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def _association_score(self, w_emb, A1_emb, A2_emb):
        """Calculate association score between word and attribute groups"""
        mean_cos_A1 = np.mean([cosine_similarity([w_emb], [a])[0][0] for a in A1_emb])
        mean_cos_A2 = np.mean([cosine_similarity([w_emb], [a])[0][0] for a in A2_emb])
        return mean_cos_A1 - mean_cos_A2
    
    def _random_effects_meta_analysis(self, effect_sizes, variances):
        """Apply random-effects model to combine effect sizes"""
        if len(effect_sizes) < 2:
            return {
                'combined_effect_size': effect_sizes[0] if effect_sizes else 0.0,
                'variance': variances[0] if variances else 0.0,
                'tau_squared': 0.0,
                'confidence_interval': (0.0, 0.0)
            }
        
        weights = [1/var if var > 0 else 0 for var in variances]
        if sum(weights) == 0:
            return {
                'combined_effect_size': 0.0,
                'variance': 0.0,
                'tau_squared': 0.0,
                'confidence_interval': (0.0, 0.0)
            }
            
        weighted_mean = np.average(effect_sizes, weights=weights)
        
        Q = sum(w * (es - weighted_mean)**2 for w, es in zip(weights, effect_sizes))
        df = len(effect_sizes) - 1
        
        if Q > df:
            sum_weights = sum(weights)
            sum_weights_squared = sum(w**2 for w in weights)
            tau_squared = max(0, (Q - df) / (sum_weights - sum_weights_squared/sum_weights))
        else:
            tau_squared = 0
        
        re_weights = [1/(var + tau_squared) if (var + tau_squared) > 0 else 0 for var in variances]
        
        if sum(re_weights) == 0:
            return {
                'combined_effect_size': 0.0,
                'variance': 0.0,
                'tau_squared': tau_squared,
                'confidence_interval': (0.0, 0.0)
            }
        
        combined_es = np.average(effect_sizes, weights=re_weights)
        combined_variance = 1/sum(re_weights) if sum(re_weights) > 0 else 0
        
        return {
            'combined_effect_size': combined_es,
            'variance': combined_variance,
            'tau_squared': tau_squared,
            'confidence_interval': self._calculate_confidence_interval(combined_es, combined_variance)
        }
    
    def _calculate_confidence_interval(self, effect_size, variance, alpha=0.05):
        """Calculate 95% confidence interval"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        margin = z_alpha * np.sqrt(variance)
        return (effect_size - margin, effect_size + margin)
    
    def compute_ceat_score_with_random_effects(self, wordlists, layeridx, language):
        """Implement proper CEAT with random-effects meta-analysis"""
        
        weat_category = wordlists.get('category', 'WEAT1')
        if weat_category in self.sentence_templates[language]:
            templates = self.sentence_templates[language][weat_category]
        else:
            templates = self.sentence_templates[language]['WEAT1']
        
        template_effect_sizes = []
        template_variances = []
        template_details = []
        
        for template_idx, template in enumerate(templates):
            T1_template_emb = self._get_template_embeddings(wordlists['targ1'], template, layeridx)
            T2_template_emb = self._get_template_embeddings(wordlists['targ2'], template, layeridx)
            A1_emb = self._get_template_embeddings(wordlists['attr1'], template, layeridx)
            A2_emb = self._get_template_embeddings(wordlists['attr2'], template, layeridx)
            
            T1_associations = [self._association_score(w, A1_emb, A2_emb) for w in T1_template_emb]
            T2_associations = [self._association_score(w, A1_emb, A2_emb) for w in T2_template_emb]
            
            all_associations = T1_associations + T2_associations
            pooled_std = np.std(all_associations)
            
            if pooled_std > 0:
                effect_size = (np.mean(T1_associations) - np.mean(T2_associations)) / pooled_std
                
                n1, n2 = len(T1_associations), len(T2_associations)
                variance = ((n1 + n2) / (n1 * n2)) + (effect_size ** 2) / (2 * (n1 + n2))
                
                template_effect_sizes.append(effect_size)
                template_variances.append(variance)
                template_details.append({
                    'template_idx': template_idx,
                    'effect_size': effect_size,
                    'variance': variance
                })
        
        combined_result = self._random_effects_meta_analysis(
            template_effect_sizes, template_variances
        )
        
        combined_result['template_details'] = template_details
        combined_result['n_templates'] = len(template_effect_sizes)
        
        total_words = len(wordlists['targ1']) + len(wordlists['targ2']) + \
                     len(wordlists['attr1']) + len(wordlists['attr2'])
        combined_result['total_sentences_generated'] = total_words * len(templates)
        combined_result['total_words'] = total_words
        
        return combined_result
    
    def _compute_permuted_ceat(self, wordlists, layeridx, language):
        """Compute CEAT with shuffled target assignments for permutation test"""
        shuffled_wordlists = wordlists.copy()
        all_targets = list(wordlists['targ1']) + list(wordlists['targ2'])
        np.random.shuffle(all_targets)
        
        mid_point = len(wordlists['targ1'])
        shuffled_wordlists['targ1'] = all_targets[:mid_point]
        shuffled_wordlists['targ2'] = all_targets[mid_point:]
        
        result = self.compute_ceat_score_with_random_effects(shuffled_wordlists, layeridx, language)
        return result
    
    def compute_ceat_significance(self, wordlists, combined_result, layeridx, language, n_permutations=50):
        """Add permutation testing for statistical significance"""
        
        observed_es = combined_result['combined_effect_size']
        
        permuted_es = []
        for _ in range(n_permutations):
            shuffled_result = self._compute_permuted_ceat(wordlists, layeridx, language)
            permuted_es.append(shuffled_result['combined_effect_size'])
        
        p_value = np.mean(np.abs(permuted_es) >= np.abs(observed_es))
        
        return {
            'p_value': p_value,
            'significant': p_value < 0.05,
            'observed_es': observed_es
        }
    
    def analyze_context_variance(self, template_details):
        """Analyze how bias varies across different contexts"""
        
        if not template_details:
            return {}
        
        effect_sizes = [r['effect_size'] for r in template_details]
        
        variance_metrics = {
            'between_context_variance': np.var(effect_sizes),
            'context_consistency': 1 - (np.std(effect_sizes) / (np.mean(np.abs(effect_sizes)) + 1e-8)),
            'max_context_difference': np.max(effect_sizes) - np.min(effect_sizes)
        }
        
        return variance_metrics

# Define this function at module level for pickling
def process_single_model(model_id, weat_categories, languages, device_id=None):
    """Process a single model - this function will be called in parallel"""
    
    # Set environment variable for this process
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"\n[Process {mp.current_process().name}] Starting evaluation of {model_id}")
    print(f"[Process {mp.current_process().name}] Using device: {device_id}")
    start_time = time.time()
    
    # Create evaluator for this process
    evaluator = EnhancedCEATEvaluator(device_id=device_id)
    weathub_loader = WEATHubLoader('iamshnoo/WEATHub', cache_dir="./datasets_cache")
    
    # Determine model type
    if model_id in BASE_MODELS:
        model_type = "base"
    else:
        model_type = "finetuned"
    
    # Load model
    if not evaluator.load_model(model_id):
        print(f"[Process {mp.current_process().name}] Failed to load {model_id}")
        return None
    
    # Get number of layers
    try:
        if hasattr(evaluator.model.config, 'num_hidden_layers'):
            num_layers = evaluator.model.config.num_hidden_layers
        elif hasattr(evaluator.model.config, 'num_transformer_layers'):
            num_layers = evaluator.model.config.num_transformer_layers
        elif hasattr(evaluator.model, 'transformer') and hasattr(evaluator.model.transformer, 'h'):
            num_layers = len(evaluator.model.transformer.h)
        else:
            num_layers = 12
    except:
        num_layers = 12
    
    print(f"[Process {mp.current_process().name}] Model {model_id} has {num_layers} layers")
    
    # Determine which languages to evaluate
    if model_type == "base":
        eval_languages = ['en']
    else:
        eval_languages = ['en', 'hi']
    
    model_results = []
    model_total_sentences = 0
    
    # Evaluate for each language and WEAT category
    for language in eval_languages:
        for weat_cat in weat_categories:
            print(f"[Process {mp.current_process().name}] Processing {model_id}: {language} - {weat_cat}")
            
            # Get word lists
            word_lists = weathub_loader.get_word_lists(language, weat_cat)
            if not word_lists:
                continue
            
            word_lists['category'] = weat_cat
            
            # Evaluate each layer
            for layer_idx in range(num_layers):
                try:
                    # Compute CEAT with random-effects meta-analysis
                    meta_result = evaluator.compute_ceat_score_with_random_effects(
                        word_lists, layer_idx, language
                    )
                    
                    # Compute statistical significance (reduced permutations for speed)
                    significance_result = evaluator.compute_ceat_significance(
                        word_lists, meta_result, layer_idx, language, n_permutations=30
                    )
                    
                    # Analyze context variance
                    variance_metrics = evaluator.analyze_context_variance(
                        meta_result.get('template_details', [])
                    )
                    
                    # Create result
                    result = {
                        'model_id': model_id,
                        'model_type': model_type,
                        'model_size_M': MODEL_SIZES.get(model_id, 0),
                        'language': language,
                        'weat_category_id': weat_cat,
                        'layer_idx': layer_idx,
                        'combined_effect_size': meta_result['combined_effect_size'],
                        'variance': meta_result['variance'],
                        'tau_squared': meta_result['tau_squared'],
                        'confidence_interval_lower': meta_result['confidence_interval'][0],
                        'confidence_interval_upper': meta_result['confidence_interval'][1],
                        'p_value': significance_result['p_value'],
                        'significant': significance_result['significant'],
                        'between_context_variance': variance_metrics.get('between_context_variance', 0),
                        'context_consistency': variance_metrics.get('context_consistency', 0),
                        'total_sentences_generated': meta_result['total_sentences_generated'],
                        'total_words': meta_result['total_words']
                    }
                    
                    model_results.append(result)
                    model_total_sentences += meta_result['total_sentences_generated']
                    
                except Exception as e:
                    print(f"[Process {mp.current_process().name}] Error at layer {layer_idx}: {e}")
                    continue
    
    # Clear model from memory
    evaluator.clear_model_cache()
    
    # Save results
    if model_results:
        df = pd.DataFrame(model_results)
        safe_model_name = model_id.replace("/", "_").replace("\\", "_")
        output_file = f"parallel_ceat_{safe_model_name}_results.csv"
        df.to_csv(output_file, index=False)
        
        elapsed_time = time.time() - start_time
        print(f"[Process {mp.current_process().name}] Completed {model_id} in {elapsed_time:.2f} seconds")
        print(f"[Process {mp.current_process().name}] Results saved to {output_file}")
        
        # Return summary
        return {
            'model_id': model_id,
            'model_type': model_type,
            'model_size_M': MODEL_SIZES.get(model_id, 0),
            'total_records': len(model_results),
            'total_sentences': model_total_sentences,
            'output_file': output_file,
            'processing_time': elapsed_time,
            'mean_effect_size': df['combined_effect_size'].mean(),
            'significant_tests': df['significant'].sum()
        }
    
    return None

def evaluate_models_parallel():
    """Main function to evaluate models with smart parallel batching"""
    
    print("\n" + "="*60)
    print("PARALLEL ENHANCED CEAT EVALUATION - FIXED VERSION")
    print("="*60)
    print(f"Available CPU cores: {mp.cpu_count()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Show GPU info if available
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print("\nGPU Information:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} MB ({gpu['memory_util']:.1f}%)")
    
    weat_categories = ['WEAT1', 'WEAT2', 'WEAT6']
    languages = ['en', 'hi']
    
    # Create smart batches for all models
    all_models = BASE_MODELS + FINETUNED_MODELS
    all_batches = create_smart_batches(all_models)
    
    print(f"\nCreated {len(all_batches)} smart batches:")
    for i, batch in enumerate(all_batches):
        total_size = sum(MODEL_SIZES.get(m, 200) for m in batch)
        print(f"  Batch {i+1}: {len(batch)} models, Total size: {total_size}M params")
        for model in batch:
            size = MODEL_SIZES.get(model, 0)
            model_type = "base" if model in BASE_MODELS else "finetuned"
            print(f"    - {model} ({size}M params, {model_type})")
    
    print("="*60)
    
    all_summaries = []
    batch_num = 1
    
    # Process each batch
    for batch in all_batches:
        print(f"\n{'='*50}")
        print(f"PROCESSING BATCH {batch_num}/{len(all_batches)}")
        print(f"Models in batch: {batch}")
        print(f"{'='*50}")
        
        batch_start_time = time.time()
        
        # Determine number of workers (max 3 as requested)
        n_gpus = torch.cuda.device_count()
        max_workers = min(len(batch), 3, n_gpus if n_gpus > 0 else mp.cpu_count() // 2)
        
        print(f"Using {max_workers} parallel workers")
        
        # Process models in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all models in the batch
            futures = []
            for i, model_id in enumerate(batch):
                # Assign GPU device if available
                device_id = i % n_gpus if n_gpus > 0 else None
                
                future = executor.submit(
                    process_single_model,
                    model_id,
                    weat_categories,
                    languages,
                    device_id
                )
                futures.append((model_id, future))
            
            # Collect results as they complete
            for model_id, future in futures:
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout per model
                    if result:
                        all_summaries.append(result)
                        print(f"✓ Completed: {model_id}")
                except Exception as e:
                    print(f"✗ Failed: {model_id} - {str(e)}")
        
        batch_elapsed = time.time() - batch_start_time
        print(f"Batch {batch_num} completed in {batch_elapsed:.2f} seconds")
        
        # Memory cleanup between batches
        gc.collect()
        torch.cuda.empty_cache()
        
        # Show GPU memory status after batch
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print("\nGPU Memory after batch:")
            for gpu in gpu_info:
                print(f"  GPU {gpu['id']}: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} MB")
        
        batch_num += 1
    
    # Save overall summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df = summary_df.sort_values('model_size_M')  # Sort by model size
        summary_df.to_csv("parallel_ceat_evaluation_summary.csv", index=False)
        
        print(f"\n{'='*60}")
        print("PARALLEL EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Total models evaluated: {len(all_summaries)}")
        print(f"✓ Summary saved to: parallel_ceat_evaluation_summary.csv")
        print(f"✓ Total sentences processed: {sum(s['total_sentences'] for s in all_summaries):,}")
        print(f"✓ Average processing time per model: {np.mean([s['processing_time'] for s in all_summaries]):.2f} seconds")
        
        print("\nModel Processing Summary (sorted by size):")
        for summary in sorted(all_summaries, key=lambda x: x['model_size_M']):
            print(f"  {summary['model_id']} ({summary['model_size_M']}M):")
            print(f"    - Time: {summary['processing_time']:.2f}s")
            print(f"    - Sentences: {summary['total_sentences']:,}")
            print(f"    - Mean Effect Size: {summary['mean_effect_size']:.4f}")
            print(f"    - Significant tests: {summary['significant_tests']}/{summary['total_records']}")
    
    print(f"\n{'='*60}")
    print("🎉 Parallel CEAT evaluation completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import GPUtil
        print(f"GPUtil available for GPU monitoring")
    except ImportError:
        print("Warning: GPUtil not installed. Install with: pip install gputil")
    
    print("\n" + "="*60)
    print("PARALLEL ENHANCED CEAT EVALUATION - FIXED VERSION")
    print("="*60)
    print("📊 Based on: https://arxiv.org/pdf/2006.03955")
    print("⚡ Smart parallel execution (max 3 models)")
    print("🎯 No two large models in same batch")
    print("📈 Random-effects meta-analysis")
    print("🔧 Using WEAT categories: WEAT1, WEAT2, WEAT6")
    print("🌍 Languages: English (en) and Hindi (hi)")
    print("="*60)
    
    # Run parallel evaluation
    evaluate_models_parallel()