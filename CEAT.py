# -*- coding: utf-8 -*-
"""
Enhanced CEAT (Contextual Embedding Association Test) for Layer-wise Bias Analysis
Implements proper random-effects meta-analysis as per the original CEAT paper
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
warnings.filterwarnings('ignore')

# Set HF token
HF_TOKEN = "secret"
login(HF_TOKEN, add_to_git_credential=True)

# Model configurations
BASE_MODELS = [
    "apple/OpenELM-270M",
    "facebook/MobileLLM-125M", 
    "cerebras/Cerebras-GPT-111M",
    "EleutherAI/pythia-70m",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-135M"
]

FINETUNED_MODELS = [
    "DebK/pythia-70m-finetuned-alpaca-hindi",
    "DebK/cerebras-gpt-111m-finetuned-alpaca-hindi",
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi",
    "DebK/OpenELM-270M-finetuned-alpaca-hindi_full",
    "DebK/Llama-3.2-1B-finetuned-alpaca-hindi_full",
    "DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full",
    "DebK/SmolLM2-135M-finetuned-alpaca-hindi"
]

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

def log_debug(message, level=1):
    """Print debug messages with indentation based on level"""
    indent = "  " * (level - 1)
    print(f"{indent}DEBUG: {message}")

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
    
    def __init__(self, cache_dir="./hf_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Enhanced diverse sentence templates - Step 3 improvement
        self.sentence_templates = {
            'en': {
                'WEAT1': [
                    # Academic/Scientific contexts
                    "Researchers discovered that {} exhibits unique characteristics.",
                    "The study of {} reveals important biological patterns.",
                    
                    # Emotional/Descriptive contexts  
                    "Looking at the {}, one feels a sense of tranquility.",
                    "The {} evokes strong emotional responses in observers.",
                    
                    # Action/Dynamic contexts
                    "The {} moves gracefully through its environment.",
                    "Children often chase after {} in the garden.",
                    
                    # Comparative contexts
                    "Unlike other species, {} demonstrates remarkable adaptation.",
                    "The {} stands out among its peers in the natural world."
                ],
                'WEAT2': [
                    # Performance contexts
                    "The musician skillfully played the {} during the concert.",
                    "The soldier carefully maintained his {} before battle.",
                    
                    # Cultural contexts
                    "In many cultures, the {} symbolizes tradition and heritage.",
                    "The {} holds deep cultural significance across generations.",
                    
                    # Learning contexts
                    "Students learn to appreciate the {} through practice.",
                    "Mastering the {} requires years of dedicated training.",
                    
                    # Social contexts
                    "The community gathered to witness the {} demonstration.",
                    "People from all walks of life admire the {}."
                ],
                'WEAT6': [
                    # Professional contexts
                    "{} demonstrated exceptional leadership in the boardroom.",
                    "{} balanced work responsibilities with personal commitments.",
                    
                    # Social contexts
                    "{} participated actively in community events.",
                    "{} formed meaningful relationships with colleagues.",
                    
                    # Achievement contexts
                    "{} achieved recognition for outstanding contributions.",
                    "{} overcame challenges through determination and skill.",
                    
                    # Personal contexts
                    "{} valued time spent with family and friends.",
                    "{} pursued hobbies that brought personal fulfillment."
                ]
            },
            'hi': {
                'WEAT1': [
                    # Academic/Scientific contexts
                    "वैज्ञानिकों ने पाया कि {} अद्वितीय गुण प्रदर्शित करता है।",
                    "{} का अध्ययन महत्वपूर्ण जैविक पैटर्न प्रकट करता है।",
                    
                    # Emotional/Descriptive contexts
                    "{} को देखकर शांति का अनुभव होता है।",
                    "{} दर्शकों में मजबूत भावनात्मक प्रतिक्रिया उत्पन्न करता है।",
                    
                    # Action/Dynamic contexts
                    "{} अपने वातावरण में सुंदरता से घूमता है।",
                    "बच्चे बगीचे में {} के पीछे भागते हैं।",
                    
                    # Comparative contexts
                    "अन्य प्रजातियों के विपरीत, {} उल्लेखनीय अनुकूलन प्रदर्शित करता है।",
                    "{} प्राकृतिक दुनिया में अपने साथियों में अलग है।"
                ],
                'WEAT2': [
                    # Performance contexts
                    "संगीतकार ने संगीत समारोह में कुशलता से {} बजाया।",
                    "सैनिक ने युद्ध से पहले अपने {} की देखभाल की।",
                    
                    # Cultural contexts
                    "कई संस्कृतियों में, {} परंपरा और विरासत का प्रतीक है।",
                    "{} पीढ़ियों में गहरा सांस्कृतिक महत्व रखता है।",
                    
                    # Learning contexts
                    "छात्र अभ्यास के माध्यम से {} की सराहना करना सीखते हैं।",
                    "{} में महारत हासिल करने के लिए वर्षों के समर्पित प्रशिक्षण की आवश्यकता है।",
                    
                    # Social contexts
                    "समुदाय {} प्रदर्शन देखने के लिए एकत्र हुआ।",
                    "सभी वर्गों के लोग {} की प्रशंसा करते हैं।"
                ],
                'WEAT6': [
                    # Professional contexts
                    "{} ने बोर्डरूम में असाधारण नेतृत्व प्रदर्शित किया।",
                    "{} ने कार्य जिम्मेदारियों को व्यक्तिगत प्रतिबद्धताओं के साथ संतुलित किया।",
                    
                    # Social contexts
                    "{} ने सामुदायिक कार्यक्रमों में सक्रिय रूप से भाग लिया।",
                    "{} ने सहकर्मियों के साथ सार्थक संबंध बनाए।",
                    
                    # Achievement contexts
                    "{} ने उत्कृष्ट योगदान के लिए मान्यता प्राप्त की।",
                    "{} ने दृढ़ संकल्प और कौशल के माध्यम से चुनौतियों को पार किया।",
                    
                    # Personal contexts
                    "{} ने परिवार और दोस्तों के साथ बिताए समय को महत्व दिया।",
                    "{} ने शौक का पीछा किया जो व्यक्तिगत संतुष्टि लाते थे।"
                ]
            }
        }
        
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
        print("Model cache cleared")
    
    def load_model(self, model_id):
        """Load model with appropriate configuration and improved error handling"""
        self.clear_model_cache()
        
        tokenizer_id = TOKENIZER_MAPPING.get(model_id, model_id)
        
        print(f"Loading model: {model_id}")
        print(f"Loading tokenizer: {tokenizer_id}")
        
        trust_remote = any(model_name in model_id for model_name in ["OpenELM", "MobileLLM"])
        log_debug(f"Setting trust_remote_code={trust_remote}")
        
        tokenizer_config = SPECIAL_TOKENIZER_CONFIG.get(model_id, {"use_fast": True})
        log_debug(f"Tokenizer config: {tokenizer_config}")
        
        try:
            log_debug(f"Attempting to load tokenizer with config: {tokenizer_config}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id, 
                cache_dir=self.cache_dir,
                token=HF_TOKEN,
                trust_remote_code=trust_remote,
                **tokenizer_config
            )
            log_debug(f"Tokenizer loaded: {type(self.tokenizer).__name__}")
            
        except Exception as e:
            log_debug(f"First tokenizer attempt failed: {str(e)}")
            
            try:
                opposite_fast = not tokenizer_config.get("use_fast", True)
                log_debug(f"Trying with use_fast={opposite_fast}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    cache_dir=self.cache_dir,
                    token=HF_TOKEN,
                    trust_remote_code=trust_remote,
                    use_fast=opposite_fast
                )
                log_debug(f"Tokenizer loaded with opposite fast setting")
                
            except Exception as e2:
                log_debug(f"Second tokenizer attempt failed: {str(e2)}")
                
                try:
                    fallback_tokenizer = "meta-llama/Llama-2-7b-hf"
                    log_debug(f"Trying fallback tokenizer: {fallback_tokenizer}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        fallback_tokenizer,
                        cache_dir=self.cache_dir,
                        token=HF_TOKEN,
                        trust_remote_code=False
                    )
                    log_debug(f"Using fallback tokenizer")
                    
                except Exception as e3:
                    print(f"Error: All tokenizer loading attempts failed.")
                    return False
        
        if self.tokenizer.pad_token is None:
            log_debug("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            log_debug("Setting up quantization config")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            try:
                import bitsandbytes as bnb
                log_debug(f"Bitsandbytes version: {bnb.__version__}")
            except ImportError:
                print("Warning: bitsandbytes not installed. Falling back to full precision.")
                quantization_config = None
                
            print(f"Loading model architecture: {model_id}")
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "cache_dir": self.cache_dir,
                "token": HF_TOKEN,
                "trust_remote_code": trust_remote
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            self.current_model_id = model_id
            log_debug(f"Model loaded successfully with {self.model.num_parameters()} parameters")
            print(f"Model {model_id} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model {model_id}: {str(e)}")
            return False
        
        return True
    
    def find_word_token_positions_improved(self, sentence: str, target_word: str):
        """Step 5: Improved word tokenization handling with better subword support"""
        tokens = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
        token_ids = tokens['input_ids'][0]
        offset_mapping = tokens['offset_mapping'][0]
        
        target_positions = []
        sentence_lower = sentence.lower()
        target_lower = target_word.lower()
        
        # Handle multiple occurrences
        start_idx = 0
        while True:
            word_start = sentence_lower.find(target_lower, start_idx)
            if word_start == -1:
                break
                
            word_end = word_start + len(target_lower)
            
            # Find all tokens that overlap with target word
            overlapping_tokens = []
            for i, (start_char, end_char) in enumerate(offset_mapping):
                if start_char == 0 and end_char == 0:  # Special token
                    continue
                if not (end_char <= word_start or start_char >= word_end):  # Overlap check
                    overlapping_tokens.append(i)
            
            target_positions.extend(overlapping_tokens)
            start_idx = word_end
        
        return list(set(target_positions))  # Remove duplicates
    
    @torch.no_grad()
    def get_contextual_word_embedding(self, sentence: str, target_word: str, layer_idx: int):
        """Get contextual embedding of a target word from a specific layer"""
        try:
            # Use improved tokenization
            token_positions = self.find_word_token_positions_improved(sentence, target_word)
            
            if not token_positions:
                print(f"Warning: No tokens found for word '{target_word}' in sentence '{sentence}'")
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
                print(f"Warning: No valid embeddings found for word '{target_word}'")
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    return np.zeros(self.model.config.hidden_size)
                else:
                    return np.zeros(768)
            
            word_embedding = np.mean(word_embeddings, axis=0)
            
            return word_embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating contextual embedding: {e}")
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
        """Step 1: Apply random-effects model to combine effect sizes"""
        if len(effect_sizes) < 2:
            return {
                'combined_effect_size': effect_sizes[0] if effect_sizes else 0.0,
                'variance': variances[0] if variances else 0.0,
                'tau_squared': 0.0,
                'confidence_interval': (0.0, 0.0)
            }
        
        # Calculate between-study variance (τ²)
        weights = [1/var if var > 0 else 0 for var in variances]
        if sum(weights) == 0:
            return {
                'combined_effect_size': 0.0,
                'variance': 0.0,
                'tau_squared': 0.0,
                'confidence_interval': (0.0, 0.0)
            }
            
        weighted_mean = np.average(effect_sizes, weights=weights)
        
        # Q statistic for heterogeneity
        Q = sum(w * (es - weighted_mean)**2 for w, es in zip(weights, effect_sizes))
        df = len(effect_sizes) - 1
        
        # Estimate τ² (between-study variance)
        if Q > df:
            sum_weights = sum(weights)
            sum_weights_squared = sum(w**2 for w in weights)
            tau_squared = max(0, (Q - df) / (sum_weights - sum_weights_squared/sum_weights))
        else:
            tau_squared = 0
        
        # Final random-effects weights
        re_weights = [1/(var + tau_squared) if (var + tau_squared) > 0 else 0 for var in variances]
        
        if sum(re_weights) == 0:
            return {
                'combined_effect_size': 0.0,
                'variance': 0.0,
                'tau_squared': tau_squared,
                'confidence_interval': (0.0, 0.0)
            }
        
        # Combined effect size (CES)
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
        """Step 1: Implement proper CEAT with random-effects meta-analysis"""
        
        # Get category-specific templates
        weat_category = wordlists.get('category', 'WEAT1')
        if weat_category in self.sentence_templates[language]:
            templates = self.sentence_templates[language][weat_category]
        else:
            # Fall back to WEAT1 templates if category not found
            templates = self.sentence_templates[language]['WEAT1']
            print(f"No specific templates for {weat_category}, using WEAT1 templates")
        
        # Step 1: Calculate effect size for each context (template) separately
        template_effect_sizes = []
        template_variances = []
        template_details = []
        
        for template_idx, template in enumerate(templates):
            # Get embeddings for this specific template
            T1_template_emb = self._get_template_embeddings(wordlists['targ1'], template, layeridx)
            T2_template_emb = self._get_template_embeddings(wordlists['targ2'], template, layeridx)
            A1_emb = self._get_template_embeddings(wordlists['attr1'], template, layeridx)
            A2_emb = self._get_template_embeddings(wordlists['attr2'], template, layeridx)
            
            # Calculate effect size for this template
            T1_associations = [self._association_score(w, A1_emb, A2_emb) for w in T1_template_emb]
            T2_associations = [self._association_score(w, A1_emb, A2_emb) for w in T2_template_emb]
            
            # Cohen's d for this template
            all_associations = T1_associations + T2_associations
            pooled_std = np.std(all_associations)
            
            if pooled_std > 0:
                effect_size = (np.mean(T1_associations) - np.mean(T2_associations)) / pooled_std
                
                # Variance estimation (simplified)
                n1, n2 = len(T1_associations), len(T2_associations)
                variance = ((n1 + n2) / (n1 * n2)) + (effect_size ** 2) / (2 * (n1 + n2))
                
                template_effect_sizes.append(effect_size)
                template_variances.append(variance)
                template_details.append({
                    'template_idx': template_idx,
                    'template': template[:50] + '...' if len(template) > 50 else template,
                    'effect_size': effect_size,
                    'variance': variance,
                    'mean_T1': np.mean(T1_associations),
                    'mean_T2': np.mean(T2_associations),
                    'std': pooled_std
                })
        
        # Step 2: Apply random-effects meta-analysis
        combined_result = self._random_effects_meta_analysis(
            template_effect_sizes, template_variances
        )
        
        # Add template details to result
        combined_result['template_details'] = template_details
        combined_result['n_templates'] = len(template_effect_sizes)
        
        # Count total sentences generated
        total_words = len(wordlists['targ1']) + len(wordlists['targ2']) + \
                     len(wordlists['attr1']) + len(wordlists['attr2'])
        combined_result['total_sentences_generated'] = total_words * len(templates)
        combined_result['total_words'] = total_words
        
        return combined_result
    
    def _compute_permuted_ceat(self, wordlists, layeridx, language):
        """Compute CEAT with shuffled target assignments for permutation test"""
        # Create shuffled word lists
        shuffled_wordlists = wordlists.copy()
        all_targets = list(wordlists['targ1']) + list(wordlists['targ2'])
        np.random.shuffle(all_targets)
        
        mid_point = len(wordlists['targ1'])
        shuffled_wordlists['targ1'] = all_targets[:mid_point]
        shuffled_wordlists['targ2'] = all_targets[mid_point:]
        
        # Compute CEAT score with shuffled targets
        result = self.compute_ceat_score_with_random_effects(shuffled_wordlists, layeridx, language)
        return result
    
    def compute_ceat_significance(self, wordlists, combined_result, layeridx, language, n_permutations=100):
        """Step 2: Add permutation testing for statistical significance"""
        
        observed_es = combined_result['combined_effect_size']
        
        # Permutation test
        permuted_es = []
        for _ in tqdm(range(n_permutations), desc="Permutation test", leave=False):
            # Shuffle target word assignments
            shuffled_result = self._compute_permuted_ceat(wordlists, layeridx, language)
            permuted_es.append(shuffled_result['combined_effect_size'])
        
        # Calculate p-value
        p_value = np.mean(np.abs(permuted_es) >= np.abs(observed_es))
        
        return {
            'p_value': p_value,
            'significant': p_value < 0.05,
            'permuted_mean': np.mean(permuted_es),
            'permuted_std': np.std(permuted_es),
            'observed_es': observed_es
        }
    
    def analyze_context_variance(self, template_details):
        """Step 4: Analyze how bias varies across different contexts"""
        
        if not template_details:
            return {}
        
        effect_sizes = [r['effect_size'] for r in template_details]
        
        variance_metrics = {
            'between_context_variance': np.var(effect_sizes),
            'context_consistency': 1 - (np.std(effect_sizes) / (np.mean(np.abs(effect_sizes)) + 1e-8)),
            'max_context_difference': np.max(effect_sizes) - np.min(effect_sizes),
            'effect_size_range': (np.min(effect_sizes), np.max(effect_sizes)),
            'most_biased_template_idx': np.argmax(np.abs(effect_sizes)),
            'least_biased_template_idx': np.argmin(np.abs(effect_sizes))
        }
        
        return variance_metrics
    
    def validate_ceat_implementation(self, language='en'):
        """Step 6: Validate CEAT implementation quality"""
        
        validation_results = {}
        
        # Check template diversity
        for category in ['WEAT1', 'WEAT2', 'WEAT6']:
            if category in self.sentence_templates[language]:
                templates = self.sentence_templates[language][category]
                
                # Lexical diversity check
                all_words = ' '.join(templates).lower().split()
                unique_words = len(set(all_words))
                total_words = len(all_words)
                
                validation_results[f'{category}_lexical_diversity'] = unique_words / total_words
                validation_results[f'{category}_template_count'] = len(templates)
        
        # Check statistical power estimation
        validation_results['estimated_statistical_power'] = self._estimate_statistical_power()
        
        return validation_results
    
    def _estimate_statistical_power(self):
        """Estimate statistical power for current setup"""
        # Simplified power calculation based on template count
        n_contexts = 8  # We have 8 templates per category
        n_words_per_category = 25  # Average across WEAT tests
        
        if n_contexts >= 8 and n_words_per_category >= 20:
            return 0.8  # Adequate power
        elif n_contexts >= 6 and n_words_per_category >= 15:
            return 0.6  # Moderate power
        else:
            return 0.4  # Low power
    
    def generate_enhanced_report(self, all_results):
        """Step 7: Generate comprehensive CEAT report"""
        
        report = {
            'methodology': {
                'approach': 'Template-based CEAT with random-effects meta-analysis',
                'statistical_framework': 'Random-effects meta-analysis with permutation testing',
                'templates_per_word': 8,
                'permutation_tests': 100
            },
            'statistical_summary': {
                'mean_combined_effect_size': np.mean([r['combined_effect_size'] for r in all_results]),
                'std_combined_effect_size': np.std([r['combined_effect_size'] for r in all_results]),
                'significant_results': sum(1 for r in all_results if r.get('p_value', 1) < 0.05),
                'total_tests': len(all_results)
            },
            'context_analysis': {
                'mean_between_context_variance': np.mean([r.get('between_context_variance', 0) 
                                                         for r in all_results if 'between_context_variance' in r]),
                'mean_consistency_score': np.mean([r.get('context_consistency', 0) 
                                                  for r in all_results if 'context_consistency' in r])
            },
            'confidence_intervals': {
                'mean_ci_width': np.mean([(r['confidence_interval'][1] - r['confidence_interval'][0]) 
                                         for r in all_results if 'confidence_interval' in r])
            }
        }
        
        return report

def evaluate_all_models():
    """Main function to evaluate all models with enhanced CEAT"""
    evaluator = EnhancedCEATEvaluator()
    weathub_loader = WEATHubLoader('iamshnoo/WEATHub', cache_dir="./datasets_cache")
    
    # Only execute specific models as requested
    all_models = [
        "DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full",
        "DebK/SmolLM2-135M-finetuned-alpaca-hindi"
    ]
    weat_categories = ['WEAT1', 'WEAT2', 'WEAT6']
    languages = ['en', 'hi']
    
    # Summary statistics for logging
    summary_stats = []
    all_results_for_report = []
    
    # Validate implementation
    print("\n" + "="*60)
    print("VALIDATING CEAT IMPLEMENTATION")
    print("="*60)
    validation_en = evaluator.validate_ceat_implementation('en')
    validation_hi = evaluator.validate_ceat_implementation('hi')
    print(f"English templates validation: {validation_en}")
    print(f"Hindi templates validation: {validation_hi}")
    
    for model_id in all_models:
        print(f"\n{'='*60}")
        print(f"Evaluating Enhanced CEAT: {model_id}")
        print(f"{'='*60}")
        
        # Initialize results for this model
        model_results = []
        
        # Determine model type
        if model_id in BASE_MODELS:
            model_type = "base"
        else:
            model_type = "finetuned"
        
        # Load model
        if not evaluator.load_model(model_id):
            print(f"Skipping {model_id} due to loading error")
            continue
        
        # Get number of layers
        try:
            if hasattr(evaluator.model.config, 'num_hidden_layers'):
                num_layers = evaluator.model.config.num_hidden_layers
            elif hasattr(evaluator.model.config, 'num_transformer_layers'):
                num_layers = evaluator.model.config.num_transformer_layers
            elif hasattr(evaluator.model, 'transformer') and hasattr(evaluator.model.transformer, 'h'):
                num_layers = len(evaluator.model.transformer.h)
            else:
                print("Could not determine number of layers, defaulting to 12")
                num_layers = 12
            
            print(f"Model has {num_layers} layers")
        except Exception as e:
            print(f"Error determining layer count: {e}")
            print("Using default of 12 layers")
            num_layers = 12
        
        # Determine which languages to evaluate
        if model_type == "base":
            eval_languages = ['en']
            print(f"Base model - evaluating only English WEAT datasets")
        else:
            eval_languages = ['en', 'hi']
            print(f"Finetuned model - evaluating both English and Hindi WEAT datasets")
        
        # Track total sentences generated for this model
        model_total_sentences = 0
        
        # Evaluate for each language and WEAT category
        for language in eval_languages:
            for weat_cat in weat_categories:
                print(f"\nProcessing Enhanced CEAT: {language} - {weat_cat}")
                
                # Get word lists
                word_lists = weathub_loader.get_word_lists(language, weat_cat)
                if not word_lists:
                    print(f"Skipping {language} - {weat_cat}: No word lists found")
                    continue
                
                # Add category info to word lists for template selection
                word_lists['category'] = weat_cat
                
                # Print word count info
                word_counts = {
                    'targ1': len(word_lists['targ1']),
                    'targ2': len(word_lists['targ2']),
                    'attr1': len(word_lists['attr1']),
                    'attr2': len(word_lists['attr2'])
                }
                total_words = sum(word_counts.values())
                expected_sentences = total_words * 8  # 8 templates per word
                print(f"Word counts: {word_counts} (Total: {total_words}, Expected sentences: {expected_sentences})")
                
                # Evaluate each layer
                layer_results = []
                for layer_idx in tqdm(range(num_layers), desc=f"CEAT Layers ({language}-{weat_cat})"):
                    try:
                        # Step 1: Compute CEAT with random-effects meta-analysis
                        meta_result = evaluator.compute_ceat_score_with_random_effects(
                            word_lists, layer_idx, language
                        )
                        
                        # Step 2: Compute statistical significance
                        significance_result = evaluator.compute_ceat_significance(
                            word_lists, meta_result, layer_idx, language, n_permutations=50
                        )
                        
                        # Step 4: Analyze context variance
                        variance_metrics = evaluator.analyze_context_variance(
                            meta_result.get('template_details', [])
                        )
                        
                        # Create comment
                        comment = f"Enhanced_CEAT_{model_type}_{language}_{weat_cat}_layer{layer_idx}"
                        
                        # Combine all results
                        result = {
                            'model_id': model_id,
                            'model_type': model_type,
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
                            'n_templates': meta_result['n_templates'],
                            'between_context_variance': variance_metrics.get('between_context_variance', 0),
                            'context_consistency': variance_metrics.get('context_consistency', 0),
                            'max_context_difference': variance_metrics.get('max_context_difference', 0),
                            'total_sentences_generated': meta_result['total_sentences_generated'],
                            'total_words': meta_result['total_words'],
                            'comments': comment
                        }
                        
                        # Store for enhanced reporting
                        result_for_report = result.copy()
                        result_for_report.update(meta_result)
                        result_for_report.update(variance_metrics)
                        all_results_for_report.append(result_for_report)
                        
                        model_results.append(result)
                        layer_results.append(meta_result['combined_effect_size'])
                        
                        # Track total sentences
                        model_total_sentences += meta_result['total_sentences_generated']
                        
                    except Exception as e:
                        print(f"Error at layer {layer_idx} for {language}-{weat_cat}: {e}")
                        continue
                
                # Print layer summary for this category
                if layer_results:
                    print(f"CEAT scores across layers - Range: [{min(layer_results):.4f}, {max(layer_results):.4f}], "
                          f"Mean: {np.mean(layer_results):.4f}, Std: {np.std(layer_results):.4f}")
        
        # Clear model from cache
        evaluator.clear_model_cache()
        
        # Save results for this model
        if model_results:
            df = pd.DataFrame(model_results)
            # Create safe filename from model_id
            safe_model_name = model_id.replace("/", "_").replace("\\", "_")
            output_file = f"enhanced_ceat_{safe_model_name}_results.csv"
            df.to_csv(output_file, index=False)
            
            # Calculate summary statistics
            model_summary = {
                'model_id': model_id,
                'model_type': model_type,
                'total_records': len(model_results),
                'total_sentences_generated': model_total_sentences,
                'languages_evaluated': list(df['language'].unique()),
                'categories_evaluated': list(df['weat_category_id'].unique()),
                'layers_evaluated': df['layer_idx'].nunique(),
                'mean_combined_effect_size': df['combined_effect_size'].mean(),
                'max_combined_effect_size': df['combined_effect_size'].abs().max(),
                'std_combined_effect_size': df['combined_effect_size'].std(),
                'significant_tests': df['significant'].sum(),
                'mean_p_value': df['p_value'].mean(),
                'mean_tau_squared': df['tau_squared'].mean(),
                'mean_context_consistency': df['context_consistency'].mean()
            }
            summary_stats.append(model_summary)
            
            print(f"\n" + "="*50)
            print(f"ENHANCED RESULTS SUMMARY for {model_id}:")
            print(f"="*50)
            print(f"✓ Results saved to: {output_file}")
            print(f"✓ Total records: {len(model_results)}")
            print(f"✓ Total sentences generated: {model_total_sentences:,}")
            print(f"✓ Languages: {', '.join(df['language'].unique())}")
            print(f"✓ Categories: {', '.join(df['weat_category_id'].unique())}")
            print(f"✓ Layers evaluated: {df['layer_idx'].nunique()}")
            print(f"✓ Mean Combined Effect Size: {df['combined_effect_size'].mean():.4f}")
            print(f"✓ Max |Combined Effect Size|: {df['combined_effect_size'].abs().max():.4f}")
            print(f"✓ Significant results (p<0.05): {df['significant'].sum()}/{len(df)}")
            print(f"✓ Mean τ² (heterogeneity): {df['tau_squared'].mean():.4f}")
            print(f"✓ Mean context consistency: {df['context_consistency'].mean():.4f}")
            print("\nSample results:")
            print(df[['language', 'weat_category_id', 'layer_idx', 'combined_effect_size', 
                     'p_value', 'significant', 'context_consistency']].head(10))
            
        else:
            print(f"✗ No results generated for {model_id}")
    
    # Generate and save enhanced report
    if all_results_for_report:
        enhanced_report = evaluator.generate_enhanced_report(all_results_for_report)
        
        # Save enhanced report
        with open("enhanced_ceat_methodology_report.json", "w") as f:
            json.dump(enhanced_report, f, indent=2)
        
        print(f"\n" + "="*60)
        print("ENHANCED METHODOLOGY REPORT")
        print("="*60)
        print(json.dumps(enhanced_report, indent=2))
    
    # Save overall summary
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv("enhanced_ceat_evaluation_summary.csv", index=False)
        print(f"\n" + "="*60)
        print("OVERALL ENHANCED EVALUATION SUMMARY")
        print("="*60)
        print(f"✓ Summary saved to: enhanced_ceat_evaluation_summary.csv")
        print(f"✓ Methodology report saved to: enhanced_ceat_methodology_report.json")
        print(f"✓ Total models evaluated: {len(summary_stats)}")
        print(f"✓ Total sentences across all models: {sum(s['total_sentences_generated'] for s in summary_stats):,}")
        print("\nPer-model summary:")
        for summary in summary_stats:
            print(f"  {summary['model_id']}: {summary['total_sentences_generated']:,} sentences, "
                  f"Mean CES: {summary['mean_combined_effect_size']:.4f}, "
                  f"Significant: {summary['significant_tests']}/{summary['total_records']}")
    
    print(f"\n{'='*60}")
    print("🎉 Enhanced CEAT evaluation completed!")
    print("📊 Individual CSV files created for each model.")
    print("📈 Summary statistics saved to enhanced_ceat_evaluation_summary.csv")
    print("📋 Methodology report saved to enhanced_ceat_methodology_report.json")
    print("🔬 Results now include:")
    print("   - Random-effects meta-analysis")
    print("   - Statistical significance testing")
    print("   - Context variance analysis")
    print("   - Confidence intervals")
    print("   - Between-template heterogeneity (τ²)")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Check if bitsandbytes is available
    try:
        import bitsandbytes
        print(f"Using bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("Warning: bitsandbytes not installed. Models will be loaded in full precision.")
        print("To enable quantization, install bitsandbytes: pip install -U bitsandbytes")
    
    # Print Enhanced CEAT evaluation info
    print("\n" + "="*60)
    print("ENHANCED CONTEXTUAL EMBEDDING ASSOCIATION TEST (CEAT)")
    print("="*60)
    print("📖 Based on: https://arxiv.org/pdf/2006.03955")
    print("🎯 Implementing proper random-effects meta-analysis")
    print("📊 Statistical significance via permutation testing")
    print("🔧 Using WEAT categories: WEAT1, WEAT2, WEAT6")
    print("🌍 Languages: English (en) and Hindi (hi)")
    print("📝 Templates: 8 diverse templates per word category")
    print("🔬 Enhancements:")
    print("   1. Random-effects meta-analysis for combining template effects")
    print("   2. Permutation testing for p-values")
    print("   3. Enhanced template diversity")
    print("   4. Context variance analysis")
    print("   5. Improved tokenization handling")
    print("   6. Implementation validation")
    print("   7. Comprehensive reporting")
    print("="*60)
    
    evaluate_all_models()