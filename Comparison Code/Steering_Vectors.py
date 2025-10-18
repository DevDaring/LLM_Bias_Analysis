# -*- coding: utf-8 -*-
"""
Steering Vectors for Bias Mitigation - Layer-wise Analysis
Implementation based on "Shifting Perspectives" paper
Compatible with the existing SEAT/WEAT research framework
"""

import os
import gc
import sys
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# Set HF token
HF_TOKEN = "Secret"
login(HF_TOKEN, add_to_git_credential=True)

# Model configurations
BASE_MODELS = [
    "apple/OpenELM-270M",
    "facebook/MobileLLM-125M",
    "cerebras/Cerebras-GPT-111M",
    "EleutherAI/pythia-70m",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B"
]

FINETUNED_MODELS = [
    "DebK/pythia-70m-finetuned-alpaca-hindi",
    "DebK/cerebras-gpt-111m-finetuned-alpaca-hindi",
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi",
    "DebK/OpenELM-270M-finetuned-alpaca-hindi_full",
    "DebK/Llama-3.2-1B-finetuned-alpaca-hindi_full",
    "DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full"
]

# Special tokenizer mappings (same as your original code)
TOKENIZER_MAPPING = {
    "apple/OpenELM-270M": "meta-llama/Llama-2-7b-hf",
    "DebK/OpenELM-270M-finetuned-alpaca-hindi_full": "meta-llama/Llama-2-7b-hf",
    "facebook/MobileLLM-125M": "meta-llama/Llama-2-7b-hf",
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi": "meta-llama/Llama-2-7b-hf"
}

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
                'WEAT1': 'original_weat', 'WEAT2': 'original_weat', 'WEAT6': 'original_weat'
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


class SteeringVectorEvaluator:
    """Steering Vector evaluation for bias mitigation"""
    
    def __init__(self, cache_dir="./hf_cache", num_train_pairs=300, num_val_pairs=100):
        self.cache_dir = cache_dir
        self.num_train_pairs = num_train_pairs
        self.num_val_pairs = num_val_pairs
        os.makedirs(cache_dir, exist_ok=True)
        
        # Stimulus prompt template as per paper
        self.stimulus_templates = {
            'en': "Consider the bias related to {axis} in the following. ",
            'hi': "{axis} से संबंधित पूर्वाग्रह पर विचार करें। "
        }
        
        # Bias axis names for stimulus prompts
        self.bias_axis_names = {
            'WEAT1': {'en': 'flowers and insects', 'hi': 'फूल और कीड़े'},
            'WEAT2': {'en': 'instruments and weapons', 'hi': 'वाद्ययंत्र और हथियार'},
            'WEAT6': {'en': 'gender and career', 'hi': 'लिंग और करियर'}
        }
        
        # Question templates for creating contrastive pairs
        self.question_templates = {
            'en': {
                'WEAT1': "Which is more pleasant: {word1} or {word2}?",
                'WEAT2': "Which is more pleasant: {word1} or {word2}?",
                'WEAT6': "Who is more career-oriented: {name1} or {name2}?"
            },
            'hi': {
                'WEAT1': "कौन अधिक सुखद है: {word1} या {word2}?",
                'WEAT2': "कौन अधिक सुखद है: {word1} या {word2}?",
                'WEAT6': "कौन अधिक करियर-उन्मुख है: {name1} या {name2}?"
            }
        }
        
        self.model = None
        self.tokenizer = None
        self.current_model_id = None
        self.steering_vectors = {}  # Store computed steering vectors
        
    def clear_model_cache(self):
        """Clears the current model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_id = None
        self.steering_vectors = {}
        gc.collect()
        torch.cuda.empty_cache()
        print("Model cache cleared")
    
    def load_model(self, model_id):
        """Load model with appropriate configuration"""
        self.clear_model_cache()
        
        tokenizer_id = TOKENIZER_MAPPING.get(model_id, model_id)
        
        print(f"Loading model: {model_id}")
        print(f"Loading tokenizer: {tokenizer_id}")
        
        trust_remote = any(model_name in model_id for model_name in ["OpenELM", "MobileLLM"])
        log_debug(f"Setting trust_remote_code={trust_remote}")
        
        tokenizer_config = SPECIAL_TOKENIZER_CONFIG.get(model_id, {"use_fast": True})
        log_debug(f"Tokenizer config: {tokenizer_config}")
        
        # Try loading tokenizer with fallback strategies
        try:
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
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    cache_dir=self.cache_dir,
                    token=HF_TOKEN,
                    trust_remote_code=trust_remote,
                    use_fast=opposite_fast
                )
            except Exception as e2:
                fallback_tokenizer = "meta-llama/Llama-2-7b-hf"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    fallback_tokenizer,
                    cache_dir=self.cache_dir,
                    token=HF_TOKEN,
                    trust_remote_code=False
                )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with fp16 precision (no quantization)
        try:
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "cache_dir": self.cache_dir,
                "token": HF_TOKEN,
                "trust_remote_code": trust_remote,
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self.current_model_id = model_id
            print(f"Model {model_id} loaded successfully in fp16 precision")
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            return False
        
        return True
    
    def create_contrastive_pairs(self, word_lists, language, weat_category):
        """
        Create contrastive prompt pairs for steering vector training
        Following paper's approach: positive (anti-stereotypical) vs negative (stereotypical)
        """
        contrastive_pairs = []
        
        # Get bias axis name for stimulus prompt
        bias_axis = self.bias_axis_names[weat_category][language]
        stimulus = self.stimulus_templates[language].format(axis=bias_axis)
        
        # Get appropriate template
        template = self.question_templates[language][weat_category]
        
        # Create pairs based on WEAT category
        if weat_category in ['WEAT1', 'WEAT2']:
            # Target words vs Attributes
            # Positive: pleasant attribute chosen, Negative: unpleasant attribute chosen
            targ_words = word_lists['targ1'] + word_lists['targ2']
            pleasant = word_lists['attr1']
            unpleasant = word_lists['attr2']
            
            for i in range(min(self.num_train_pairs + self.num_val_pairs, 
                              len(targ_words) * min(len(pleasant), len(unpleasant)))):
                targ_idx = i % len(targ_words)
                attr_idx = i % min(len(pleasant), len(unpleasant))
                
                target = targ_words[targ_idx]
                pleasant_word = pleasant[attr_idx]
                unpleasant_word = unpleasant[attr_idx]
                
                question = template.format(word1=target, word2=pleasant_word)
                positive_prompt = stimulus + question + f" Answer: {pleasant_word}"
                
                question = template.format(word1=target, word2=unpleasant_word)
                negative_prompt = stimulus + question + f" Answer: {unpleasant_word}"
                
                contrastive_pairs.append({
                    'positive': positive_prompt,
                    'negative': negative_prompt
                })
                
        elif weat_category == 'WEAT6':
            # Names vs Career/Family attributes
            # Positive: equal/unbiased association, Negative: stereotypical association
            male_names = word_lists['targ1']
            female_names = word_lists['targ2']
            career = word_lists['attr1']
            family = word_lists['attr2']
            
            for i in range(min(self.num_train_pairs + self.num_val_pairs,
                              min(len(male_names), len(female_names)) * min(len(career), len(family)))):
                male_idx = i % len(male_names)
                female_idx = i % len(female_names)
                attr_idx = i % min(len(career), len(family))
                
                male_name = male_names[male_idx]
                female_name = female_names[female_idx]
                career_word = career[attr_idx]
                
                # Positive: anti-stereotypical (both equally career-oriented)
                question = template.format(name1=male_name, name2=female_name)
                positive_prompt = stimulus + question + " Answer: Both are equally career-oriented"
                
                # Negative: stereotypical (male more career-oriented)
                negative_prompt = stimulus + question + f" Answer: {male_name}"
                
                contrastive_pairs.append({
                    'positive': positive_prompt,
                    'negative': negative_prompt
                })
        
        return contrastive_pairs
    
    @torch.no_grad()
    def get_hidden_state(self, text: str, layer_idx: int):
        """Extract hidden state at specific layer for given text"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_idx]
            
            # Get last token representation (following paper)
            last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
            last_token_idx = last_token_idx.item()  # Convert to Python int
            last_token_hidden = hidden_state[0, last_token_idx, :]
            
            return last_token_hidden.float().cpu().numpy()
        except Exception as e:
            print(f"Error extracting hidden state: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_steering_vector(self, contrastive_pairs, layer_idx, split='train'):
        """
        Compute steering vector using PCA on activation differences
        Following paper's LAT (Linear Artificial Tomography) approach
        """
        total_pairs = len(contrastive_pairs)
        # Adjust split if we don't have enough pairs
        if total_pairs < self.num_train_pairs + self.num_val_pairs:
            # Use 75% for train, 25% for validation
            train_size = int(total_pairs * 0.75)
        else:
            train_size = self.num_train_pairs
        
        if split == 'train':
            pairs = contrastive_pairs[:train_size]
        else:  # validation
            pairs = contrastive_pairs[train_size:]
        
        activation_diffs = []
        
        print(f"  Computing activation differences for layer {layer_idx}...")
        for pair in tqdm(pairs, desc=f"  Processing pairs", leave=False):
            # Get hidden states for positive and negative prompts
            h_positive = self.get_hidden_state(pair['positive'], layer_idx)
            h_negative = self.get_hidden_state(pair['negative'], layer_idx)
            
            if h_positive is not None and h_negative is not None:
                # Compute difference: h^(t,+) - h^(t,-)
                diff = h_positive - h_negative
                activation_diffs.append(diff)
        
        if len(activation_diffs) == 0:
            return None, None
        
        # Stack into matrix X_{l,t}
        X = np.array(activation_diffs)
        
        # Apply PCA and extract first principal component (steering vector)
        pca = PCA(n_components=1)
        pca.fit(X)
        steering_vector = pca.components_[0]
        
        # Also return the transformed data for separability analysis
        X_transformed = pca.transform(X)
        
        return steering_vector, X_transformed
    
    def measure_linear_separability(self, contrastive_pairs, layer_idx, split='validation'):
        """
        Measure linear separability using Logistic Regression
        Following paper's approach to identify optimal layers
        """
        total_pairs = len(contrastive_pairs)
        # Adjust split if we don't have enough pairs
        if total_pairs < self.num_train_pairs + self.num_val_pairs:
            # Use 75% for train, 25% for validation
            train_size = int(total_pairs * 0.75)
        else:
            train_size = self.num_train_pairs
        
        if split == 'train':
            pairs = contrastive_pairs[:train_size]
        else:  # validation
            pairs = contrastive_pairs[train_size:]
        
        X_positive = []
        X_negative = []
        
        for pair in pairs:
            h_pos = self.get_hidden_state(pair['positive'], layer_idx)
            h_neg = self.get_hidden_state(pair['negative'], layer_idx)
            
            if h_pos is not None and h_neg is not None:
                X_positive.append(h_pos)
                X_negative.append(h_neg)
        
        if len(X_positive) == 0 or len(X_negative) == 0:
            return 0.0
        
        # Prepare data for classification
        X = np.vstack([X_positive, X_negative])
        y = np.array([1] * len(X_positive) + [0] * len(X_negative))
        
        # Apply 2-component PCA (as in paper's visualization)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Train logistic regression classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_pca, y)
        
        # Measure accuracy as proxy for linear separability
        y_pred = clf.predict(X_pca)
        accuracy = accuracy_score(y, y_pred)
        
        return accuracy
    
    @torch.no_grad()
    def apply_steering_vector(self, text: str, steering_vector: np.ndarray, 
                             layer_idx: int, coefficient: float = 1.0):
        """
        Apply steering vector at inference time
        Modifies hidden states: h' = h + λ * w
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # FIX: Normalize and scale steering vector to prevent destroying representations
            steering_normalized = steering_vector / (np.linalg.norm(steering_vector) + 1e-8)
            steering_normalized = steering_normalized * 0.5  # Scale down
            
            # FIX: Ensure proper dtype conversion to fp16
            steering_vector_tensor = torch.from_numpy(steering_normalized).float().to(self.model.device)
            steering_vector_tensor = steering_vector_tensor.half()  # Convert to fp16
            
            def steering_hook(module, input, output):
                # output is tuple: (hidden_states, ...)
                hidden_states = output[0] if isinstance(output, tuple) else output
                
                # Apply steering: h' = h + λ * w
                # Broadcast steering vector to all tokens
                steering_addition = coefficient * steering_vector_tensor.unsqueeze(0).unsqueeze(0)
                # FIX: Ensure same dtype as hidden_states
                steering_addition = steering_addition.to(hidden_states.dtype)
                modified_hidden = hidden_states + steering_addition
                
                if isinstance(output, tuple):
                    return (modified_hidden,) + output[1:]
                else:
                    return modified_hidden
            
            # Register hook at target layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # For models like Llama, Qwen, MobileLLM
                hook_handle = self.model.model.layers[layer_idx].register_forward_hook(steering_hook)
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
                # For models like OpenELM
                hook_handle = self.model.transformer.layers[layer_idx].register_forward_hook(steering_hook)
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # For models like GPT, Cerebras
                hook_handle = self.model.transformer.h[layer_idx].register_forward_hook(steering_hook)
            elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
                # For Pythia
                hook_handle = self.model.gpt_neox.layers[layer_idx].register_forward_hook(steering_hook)
            else:
                print(f"Warning: Unknown model architecture for {self.current_model_id}, cannot apply steering vector")
                print(f"Available attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')][:10]}")
                return None
            
            # Forward pass with steering
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Remove hook
            hook_handle.remove()
            
            # Get final hidden state
            last_hidden = outputs.hidden_states[-1]
            last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
            last_token_idx = last_token_idx.item()  # FIX: Convert to Python int
            final_representation = last_hidden[0, last_token_idx, :]
            
            return final_representation.float().cpu().numpy()
            
        except Exception as e:
            print(f"Error applying steering vector: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_bias_reduction(self, word_lists, steering_vector, layer_idx, 
                                language, weat_category, coefficient=1.0):
        """
        Evaluate bias reduction after applying steering vector
        Compute bias score similar to WEAT/SEAT
        FIX: Use more samples and training set, properly reshape arrays
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # FIX: Use training set instead of validation set (more reliable)
        # Create test prompts
        test_pairs = self.create_contrastive_pairs(word_lists, language, weat_category)
        
        # FIX: Use first 100 pairs from training set (not validation)
        test_pairs = test_pairs[:min(100, len(test_pairs))]
        
        bias_scores_baseline = []
        bias_scores_steered = []
        
        for pair in test_pairs:
            # Baseline (no steering)
            h_pos_baseline = self.get_hidden_state(pair['positive'], layer_idx)
            h_neg_baseline = self.get_hidden_state(pair['negative'], layer_idx)
            
            if h_pos_baseline is not None and h_neg_baseline is not None:
                # FIX: Reshape arrays properly for cosine_similarity
                h_pos_baseline = h_pos_baseline.reshape(1, -1)
                h_neg_baseline = h_neg_baseline.reshape(1, -1)
                sim_baseline = cosine_similarity(h_pos_baseline, h_neg_baseline)[0][0]
                bias_scores_baseline.append(abs(sim_baseline))
            
            # With steering vector
            h_pos_steered = self.apply_steering_vector(pair['positive'], steering_vector, 
                                                       layer_idx, coefficient)
            h_neg_steered = self.apply_steering_vector(pair['negative'], steering_vector,
                                                       layer_idx, coefficient)
            
            if h_pos_steered is not None and h_neg_steered is not None:
                # FIX: Reshape arrays properly
                h_pos_steered = h_pos_steered.reshape(1, -1)
                h_neg_steered = h_neg_steered.reshape(1, -1)
                sim_steered = cosine_similarity(h_pos_steered, h_neg_steered)[0][0]
                bias_scores_steered.append(abs(sim_steered))
        
        # Return average bias scores
        avg_bias_baseline = np.mean(bias_scores_baseline) if bias_scores_baseline else 0.0
        avg_bias_steered = np.mean(bias_scores_steered) if bias_scores_steered else 0.0
        
        # Bias reduction percentage
        if avg_bias_baseline > 0:
            reduction = ((avg_bias_baseline - avg_bias_steered) / avg_bias_baseline) * 100
        else:
            reduction = 0.0
        
        return {
            'baseline_bias': avg_bias_baseline,
            'steered_bias': avg_bias_steered,
            'reduction_percent': reduction
        }


def evaluate_all_models_steering():
    """Main evaluation function for steering vectors"""
    evaluator = SteeringVectorEvaluator(num_train_pairs=300, num_val_pairs=100)
    weathub_loader = WEATHubLoader('iamshnoo/WEATHub', cache_dir="./datasets_cache")
    
    all_models = BASE_MODELS + FINETUNED_MODELS
    weat_categories = ['WEAT1', 'WEAT2', 'WEAT6']
    
    all_results = []
    
    for model_id in all_models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_id}")
        print(f"{'='*60}")
        
        # Track results for current model only
        model_results = []
        
        # Determine model type
        if model_id in BASE_MODELS:
            model_type = "base"
            eval_languages = ['en']
        else:
            model_type = "finetuned"
            eval_languages = ['en', 'hi']
        
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
            else:
                num_layers = 12
            print(f"Model has {num_layers} layers")
        except:
            num_layers = 12
        
        # Evaluate for each language and WEAT category
        for language in eval_languages:
            for weat_cat in weat_categories:
                print(f"\n--- Processing: {language} - {weat_cat} ---")
                
                # Get word lists
                word_lists = weathub_loader.get_word_lists(language, weat_cat)
                if not word_lists:
                    continue
                
                # Create contrastive pairs
                print("Creating contrastive pairs...")
                contrastive_pairs = evaluator.create_contrastive_pairs(
                    word_lists, language, weat_cat
                )
                print(f"Created {len(contrastive_pairs)} contrastive pairs")
                
                # Phase 1: Measure linear separability across all layers
                print("\nPhase 1: Measuring linear separability across layers...")
                separability_scores = []
                
                for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
                    sep_score = evaluator.measure_linear_separability(
                        contrastive_pairs, layer_idx, split='validation'
                    )
                    separability_scores.append(sep_score)
                    
                    model_results.append({
                        'model_id': model_id,
                        'model_type': model_type,
                        'language': language,
                        'weat_category': weat_cat,
                        'layer_idx': layer_idx,
                        'phase': 'separability_analysis',
                        'linear_separability': sep_score,
                        'steering_vector_applied': False,
                        'coefficient': 0.0,
                        'baseline_bias': None,
                        'steered_bias': None,
                        'bias_reduction_percent': None,
                        'comments': f"LinearSep_{model_type}_{language}_{weat_cat}_layer{layer_idx}"
                    })
                
                # Identify optimal layers (top 2 with highest separability)
                sorted_layers = np.argsort(separability_scores)[::-1]
                optimal_layers = sorted_layers[:2].tolist()
                print(f"\nOptimal layers identified: {optimal_layers}")
                print(f"Separability scores: {[separability_scores[i] for i in optimal_layers]}")
                
                # Phase 2: Compute steering vectors for optimal layers
                print("\nPhase 2: Computing steering vectors for optimal layers...")
                
                for layer_idx in optimal_layers:
                    print(f"\nProcessing layer {layer_idx}...")
                    
                    # Compute steering vector
                    steering_vector, X_transformed = evaluator.compute_steering_vector(
                        contrastive_pairs, layer_idx, split='train'
                    )
                    
                    if steering_vector is None:
                        print(f"Failed to compute steering vector for layer {layer_idx}")
                        continue
                    
                    print(f"Steering vector computed (shape: {steering_vector.shape})")
                    
                    # Phase 3: Evaluate with different coefficients
                    print(f"Phase 3: Evaluating coefficients for layer {layer_idx}...")
                    
                    # Test coefficients from paper: -2.0 to 2.0
                    test_coefficients = [-2.0, -1.0, 0.0, 1.0, 1.6, 2.0]
                    
                    for coef in test_coefficients:
                        print(f"  Testing coefficient: {coef}")
                        
                        if coef == 0.0:
                            # Baseline (no steering)
                            bias_results = evaluator.evaluate_bias_reduction(
                                word_lists, steering_vector, layer_idx,
                                language, weat_cat, coefficient=0.0
                            )
                            steering_applied = False
                        else:
                            # With steering
                            bias_results = evaluator.evaluate_bias_reduction(
                                word_lists, steering_vector, layer_idx,
                                language, weat_cat, coefficient=coef
                            )
                            steering_applied = True
                        
                        model_results.append({
                            'model_id': model_id,
                            'model_type': model_type,
                            'language': language,
                            'weat_category': weat_cat,
                            'layer_idx': layer_idx,
                            'phase': 'bias_mitigation',
                            'linear_separability': separability_scores[layer_idx],
                            'steering_vector_applied': steering_applied,
                            'coefficient': coef,
                            'baseline_bias': bias_results['baseline_bias'],
                            'steered_bias': bias_results['steered_bias'],
                            'bias_reduction_percent': bias_results['reduction_percent'],
                            'comments': f"SteeringVec_{model_type}_{language}_{weat_cat}_layer{layer_idx}_coef{coef}"
                        })
                        
                        print(f"    Baseline bias: {bias_results['baseline_bias']:.4f}")
                        print(f"    Steered bias: {bias_results['steered_bias']:.4f}")
                        print(f"    Reduction: {bias_results['reduction_percent']:.2f}%")
        
        # Clear model
        evaluator.clear_model_cache()
        
        # Save intermediate results for THIS model only
        if model_results:
            df = pd.DataFrame(model_results)
            safe_model_name = model_id.replace("/", "_").replace("\\", "_")
            output_file = f"steering_vectors_{safe_model_name}_results.csv"
            df.to_csv(output_file, index=False)
            print(f"\nIntermediate results saved to {output_file}")
            
            # Add to combined results
            all_results.extend(model_results)
    
    # Save final combined results
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv("steering_vectors_ALL_MODELS_results.csv", index=False)
        print(f"\n{'='*60}")
        print("All results saved to steering_vectors_ALL_MODELS_results.csv")
        print(f"Total records: {len(all_results)}")
        print(f"{'='*60}")
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        
        # Bias reduction summary
        mitigation_df = final_df[final_df['phase'] == 'bias_mitigation']
        if len(mitigation_df) > 0:
            print("\nBias Reduction Summary:")
            print(f"Average baseline bias: {mitigation_df['baseline_bias'].mean():.4f}")
            print(f"Average steered bias: {mitigation_df['steered_bias'].mean():.4f}")
            print(f"Average reduction: {mitigation_df['bias_reduction_percent'].mean():.2f}%")
            
            # Best results per category
            print("\nBest Results per Category:")
            for weat_cat in weat_categories:
                cat_data = mitigation_df[mitigation_df['weat_category'] == weat_cat]
                if len(cat_data) > 0:
                    best_idx = cat_data['bias_reduction_percent'].idxmax()
                    best_result = cat_data.loc[best_idx]
                    print(f"\n{weat_cat}:")
                    print(f"  Model: {best_result['model_id']}")
                    print(f"  Layer: {best_result['layer_idx']}")
                    print(f"  Coefficient: {best_result['coefficient']}")
                    print(f"  Reduction: {best_result['bias_reduction_percent']:.2f}%")


if __name__ == "__main__":
    print("="*60)
    print("STEERING VECTORS FOR BIAS MITIGATION")
    print("Implementation based on 'Shifting Perspectives' paper")
    print("="*60)
    
    # Check dependencies
    try:
        import bitsandbytes
        print(f"Using bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("Warning: bitsandbytes not installed - using full precision")
    
    # Run evaluation
    evaluate_all_models_steering()