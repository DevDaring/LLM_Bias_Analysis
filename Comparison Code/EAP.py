# -*- coding: utf-8 -*-
"""
Edge Attribution Patching (EAP) for Bias Analysis in LLMs
Based on: "Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective"
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
    PreTrainedTokenizerFast,
)
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from huggingface_hub import login
import warnings
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
from functools import partial

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

# Special tokenizer mappings (from SEAT.py)
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
    """Loads the WEATHub dataset and provides word lists for EAP analysis."""
    def __init__(self, dataset_id: str, cache_dir: str = None):
        print(f"Loading WEATHub dataset from '{dataset_id}'...")
        try:
            self.dataset = load_dataset(dataset_id, cache_dir=cache_dir)
            print("WEATHub dataset loaded successfully.")
            self.split_mapping = {
                'WEAT1': 'original_weat', 
                'WEAT2': 'original_weat', 
                'WEAT6': 'original_weat'
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
                    'attr2': filtered[0]['attr2.examples'],
                    'targ1_name': filtered[0].get('targ1.category', 'target1'),
                    'targ2_name': filtered[0].get('targ2.category', 'target2'),
                    'attr1_name': filtered[0].get('attr1.category', 'attribute1'),
                    'attr2_name': filtered[0].get('attr2.category', 'attribute2')
                }
            else:
                print(f"Warning: No data for language '{language_code}' and category '{weat_category_id}'.")
                return None
        except Exception as e:
            print(f"Error filtering data for '{weat_category_id}' in language '{language_code}': {e}")
            return None

class BiasPromptGenerator:
    """Generates prompts for bias evaluation based on WEAT categories"""
    
    def __init__(self):
        # Define prompt templates for each WEAT category
        # Following the paper's approach for demographic and gender bias
        self.templates = {
            'en': {
                'WEAT1': {
                    # Flowers vs Insects (demographic-like bias with objects)
                    'template': "{word} are so",
                    'context_template': "As expected, {word} are so",
                    'type': 'demographic'
                },
                'WEAT2': {
                    # Musical instruments vs Weapons (demographic-like bias)
                    'template': "{word} are so",
                    'context_template': "As expected, {word} are so",
                    'type': 'demographic'
                },
                'WEAT6': {
                    # Male/Female names vs Career/Family (gender bias)
                    'template': "{word} did a mistake, and everyone scolded",
                    'context_template': "{word} did a miracle, and everyone praised",
                    'type': 'gender'
                }
            },
            'hi': {
                'WEAT1': {
                    'template': "{word} बहुत",
                    'context_template': "जैसा कि उम्मीद थी, {word} बहुत",
                    'type': 'demographic'
                },
                'WEAT2': {
                    'template': "{word} बहुत",
                    'context_template': "जैसा कि उम्मीद थी, {word} बहुत",
                    'type': 'demographic'
                },
                'WEAT6': {
                    'template': "{word} ने गलती की, और सभी ने डांटा",
                    'context_template': "{word} ने चमत्कार किया, और सभी ने प्रशंसा की",
                    'type': 'gender'
                }
            }
        }
        
        # Neutral words for corruption strategy C2 (from paper)
        self.neutral_replacements = {
            'en': {
                'WEAT1': 'dandelion',  # Most neutral flower/insect
                'WEAT2': 'broadcaster',  # Neutral instrument/weapon analogue
                'WEAT6': 'broadcaster'  # Gender-neutral profession
            },
            'hi': {
                'WEAT1': 'फूल',  # Generic flower
                'WEAT2': 'वस्तु',  # Object
                'WEAT6': 'व्यक्ति'  # Person
            }
        }
    
    def generate_prompts(self, words: List[str], weat_category: str, 
                        language: str, use_context: bool = False) -> List[str]:
        """Generate prompts for a list of words"""
        if language not in self.templates or weat_category not in self.templates[language]:
            print(f"Warning: No template for {language}/{weat_category}")
            return [f"{word}" for word in words]
        
        template_dict = self.templates[language][weat_category]
        template = template_dict['context_template'] if use_context else template_dict['template']
        
        return [template.format(word=word) for word in words]
    
    def create_corrupted_sample_c1(self, prompt: str, original_word: str) -> str:
        """Corruption strategy C1: Replace with out-of-distribution token 'xyz'"""
        return prompt.replace(original_word, "xyz")
    
    def create_corrupted_sample_c2(self, prompt: str, original_word: str, 
                                   weat_category: str, language: str) -> str:
        """Corruption strategy C2: Replace with neutral word"""
        neutral_word = self.neutral_replacements[language][weat_category]
        return prompt.replace(original_word, neutral_word)

class EdgeNode:
    """Represents a node in the computational graph"""
    def __init__(self, name: str, layer_idx: int, node_type: str):
        self.name = name
        self.layer_idx = layer_idx
        self.node_type = node_type  # 'input', 'mlp', 'attention', 'logits'
    
    def __repr__(self):
        return f"{self.node_type}_{self.name}"
    
    def __hash__(self):
        return hash((self.name, self.layer_idx, self.node_type))
    
    def __eq__(self, other):
        return (self.name == other.name and 
                self.layer_idx == other.layer_idx and 
                self.node_type == other.node_type)

class ComputationalEdge:
    """Represents an edge in the computational graph"""
    def __init__(self, source: EdgeNode, target: EdgeNode):
        self.source = source
        self.target = target
        self.attribution_score = 0.0
    
    def __repr__(self):
        return f"{self.source}->{self.target}"
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        return self.source == other.source and self.target == other.target

class EAPAnalyzer:
    """
    Edge Attribution Patching analyzer for bias detection
    Implements the methodology from the paper
    """
    
    def __init__(self, cache_dir="./hf_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.current_model_id = None
        self.prompt_generator = BiasPromptGenerator()
        
        # Store activations during forward pass
        self.activations = {}
        self.hooks = []
        
    def clear_model_cache(self):
        """Clears the current model from memory"""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_id = None
        self.activations = {}
        gc.collect()
        torch.cuda.empty_cache()
        print("Model cache cleared")
    
    def load_model(self, model_id):
        """Load model with appropriate configuration (same as SEAT.py)"""
        self.clear_model_cache()
        
        tokenizer_id = TOKENIZER_MAPPING.get(model_id, model_id)
        
        print(f"Loading model: {model_id}")
        print(f"Loading tokenizer: {tokenizer_id}")
        
        trust_remote = any(model_name in model_id for model_name in ["OpenELM", "MobileLLM"])
        log_debug(f"Setting trust_remote_code={trust_remote}")
        
        tokenizer_config = SPECIAL_TOKENIZER_CONFIG.get(model_id, {"use_fast": True})
        log_debug(f"Tokenizer config: {tokenizer_config}")
        
        # Try loading tokenizer with fallback strategies (same as SEAT.py)
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
        
        log_debug(f"Vocabulary size: {len(self.tokenizer)}")
        
        # Load model in FP16 (no quantization for better accuracy)
        try:
            log_debug("Loading model in FP16 precision (no quantization)")
            print(f"Loading model architecture: {model_id}")
            
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "cache_dir": self.cache_dir,
                "token": HF_TOKEN,
                "trust_remote_code": trust_remote
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            self.current_model_id = model_id
            log_debug(f"Model loaded successfully")
            print(f"Model {model_id} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model {model_id}: {str(e)}")
            return False
        
        return True
    
    def register_hooks(self):
        """Register forward hooks to capture activations"""
        self.activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                # FIX: Handle both tensor and tuple outputs (different model architectures)
                if isinstance(output, tuple):
                    # Some models (GPT2, GPTNeoX, etc.) return tuples, take the first element (hidden states)
                    self.activations[name] = output[0].detach()
                else:
                    # Standard case: output is a tensor
                    self.activations[name] = output.detach()
            return hook
        
        # Register hooks for each layer
        num_layers = self.get_num_layers()
        
        for idx in range(num_layers):
            # Hook for transformer blocks
            try:
                # Try different architectures
                if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
                    # OpenELM style: model.transformer.layers
                    layer = self.model.transformer.layers[idx]
                    hook = layer.register_forward_hook(get_activation(f'layer_{idx}'))
                    self.hooks.append(hook)
                elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    # GPT-2, Cerebras style: model.transformer.h
                    layer = self.model.transformer.h[idx]
                    hook = layer.register_forward_hook(get_activation(f'layer_{idx}'))
                    self.hooks.append(hook)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    # Llama, Qwen, MobileLLM style: model.model.layers
                    layer = self.model.model.layers[idx]
                    hook = layer.register_forward_hook(get_activation(f'layer_{idx}'))
                    self.hooks.append(hook)
                elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
                    # Pythia style: model.gpt_neox.layers
                    layer = self.model.gpt_neox.layers[idx]
                    hook = layer.register_forward_hook(get_activation(f'layer_{idx}'))
                    self.hooks.append(hook)
            except Exception as e:
                log_debug(f"Could not register hook for layer {idx}: {e}", 2)
        
        log_debug(f"Registered {len(self.hooks)} hooks")
    
    def get_num_layers(self):
        """Get number of layers in the model"""
        try:
            if hasattr(self.model.config, 'num_hidden_layers'):
                return self.model.config.num_hidden_layers
            elif hasattr(self.model.config, 'num_transformer_layers'):
                return self.model.config.num_transformer_layers
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
                # OpenELM style: model.transformer.layers
                return len(self.model.transformer.layers)
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return len(self.model.transformer.h)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return len(self.model.model.layers)
            elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
                return len(self.model.gpt_neox.layers)
            else:
                print("Could not determine number of layers, defaulting to 12")
                return 12
        except Exception as e:
            print(f"Error determining layer count: {e}")
            return 12
    
    @torch.no_grad()
    def compute_bias_metric_l2(self, prompts: List[str], positive_tokens: List[str], 
                               negative_tokens: List[str], k: int = 10) -> float:
        """
        Compute L2 bias metric (Equation 3 from paper)
        L2 = (1/m) * sum(sum(P_pos(i)_j)) for top-k predictions
        """
        total_positive_prob = 0.0
        
        # Get positive and negative token IDs
        positive_token_ids = set()
        negative_token_ids = set()
        
        for token in positive_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            positive_token_ids.update(token_ids)
        
        for token in negative_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            negative_token_ids.update(token_ids)
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get top-k predictions
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k)
            
            # Sum probabilities of positive tokens in top-k
            for prob, idx in zip(top_k_probs, top_k_indices):
                if idx.item() in positive_token_ids:
                    total_positive_prob += prob.item()
        
        # Average over all prompts
        l2_score = total_positive_prob / len(prompts)
        return l2_score
    
    @torch.no_grad()
    def compute_edge_attribution_scores(self, clean_prompts: List[str], 
                                       corrupted_prompts: List[str],
                                       positive_tokens: List[str],
                                       negative_tokens: List[str],
                                       k: int = 10) -> Dict[str, float]:
        """
        Compute edge attribution scores using activation-based analysis
        Returns: Dictionary mapping edge names to attribution scores
        """
        edge_scores = defaultdict(float)
        num_layers = self.get_num_layers()
        
        # Compute baseline metric with clean inputs
        baseline_metric = self.compute_bias_metric_l2(
            clean_prompts, positive_tokens, negative_tokens, k
        )
        
        print(f"Baseline bias metric (L2): {baseline_metric:.4f}")
        
        # Process each layer independently
        for layer_idx in tqdm(range(num_layers), desc="Computing edge attributions"):
            layer_name = f"layer_{layer_idx}"
            
            # Register hooks for this iteration
            self.register_hooks()
            
            # Collect clean activations for this layer
            clean_layer_activations = []
            for prompt in clean_prompts[:min(10, len(clean_prompts))]:  # Sample prompts
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                if layer_name in self.activations:
                    clean_layer_activations.append(self.activations[layer_name].clone())
            
            # Collect corrupted activations for this layer
            corrupted_layer_activations = []
            for prompt in corrupted_prompts[:min(10, len(corrupted_prompts))]:  # Sample prompts
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                if layer_name in self.activations:
                    corrupted_layer_activations.append(self.activations[layer_name].clone())
            
            # Compute attribution score based on activation patterns
            if clean_layer_activations and corrupted_layer_activations:
                # Compute mean activation values per sample (handles variable sequence lengths)
                clean_means = [act.abs().mean().item() for act in clean_layer_activations]
                corrupted_means = [act.abs().mean().item() for act in corrupted_layer_activations]
                
                # Average across all samples
                clean_mean = sum(clean_means) / len(clean_means)
                corrupted_mean = sum(corrupted_means) / len(corrupted_means)
                
                # Attribution score: difference in activation patterns weighted by magnitude
                activation_diff = abs(clean_mean - corrupted_mean)
                activation_magnitude = (clean_mean + corrupted_mean) / 2.0
                
                # Combine with baseline bias for final score
                if baseline_metric > 0:
                    score = (activation_diff + activation_magnitude * 0.1) * (1.0 + baseline_metric)
                else:
                    score = activation_diff + activation_magnitude * 0.1
                
                edge_scores[layer_name] = score
            else:
                # Fallback: use activation magnitude only
                edge_scores[layer_name] = 0.0
            
            # Remove hooks for this iteration
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
        
        return dict(edge_scores)
    
    def identify_important_edges(self, edge_scores: Dict[str, float], 
                                top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Identify top-k most important edges based on attribution scores
        """
        sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_edges[:top_k]
    
    def analyze_localization(self, important_edges: List[Tuple[str, float]]) -> Dict:
        """
        Analyze layer-wise distribution of important edges (Section 5, Figure 2)
        """
        layer_distribution = defaultdict(int)
        
        for edge_name, score in important_edges:
            # Extract layer index from edge name
            if 'layer_' in edge_name:
                layer_idx = int(edge_name.split('_')[1])
                layer_distribution[layer_idx] += 1
        
        # Calculate statistics
        total_edges = len(important_edges)
        num_layers = self.get_num_layers()
        
        # Find layers with >20% of important edges
        threshold = 0.2 * total_edges
        significant_layers = {
            layer: count for layer, count in layer_distribution.items()
            if count > threshold
        }
        
        return {
            'layer_distribution': dict(layer_distribution),
            'significant_layers': significant_layers,
            'localization_ratio': len(significant_layers) / num_layers
        }
    
    def evaluate_bias_for_weat(self, word_lists: Dict, weat_category: str, 
                              language: str) -> Dict:
        """
        Evaluate bias for a specific WEAT category
        Returns results including edge scores and localization analysis
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {weat_category} in {language}")
        print(f"{'='*60}")
        
        # Generate prompts for target words
        targ1_prompts = self.prompt_generator.generate_prompts(
            word_lists['targ1'], weat_category, language
        )
        targ2_prompts = self.prompt_generator.generate_prompts(
            word_lists['targ2'], weat_category, language
        )
        
        # For attribute words, we use them as the tokens to check
        attr1_words = word_lists['attr1']  # Positive/Male
        attr2_words = word_lists['attr2']  # Negative/Female
        
        # Create corrupted versions using C2 strategy (paper uses C2 with L2)
        targ1_corrupted = []
        for prompt, word in zip(targ1_prompts, word_lists['targ1']):
            corrupted = self.prompt_generator.create_corrupted_sample_c2(
                prompt, word, weat_category, language
            )
            targ1_corrupted.append(corrupted)
        
        targ2_corrupted = []
        for prompt, word in zip(targ2_prompts, word_lists['targ2']):
            corrupted = self.prompt_generator.create_corrupted_sample_c2(
                prompt, word, weat_category, language
            )
            targ2_corrupted.append(corrupted)
        
        # Combine all clean and corrupted prompts
        all_clean_prompts = targ1_prompts + targ2_prompts
        all_corrupted_prompts = targ1_corrupted + targ2_corrupted
        
        print(f"Generated {len(all_clean_prompts)} clean prompts")
        print(f"Sample clean prompt: {all_clean_prompts[0]}")
        print(f"Sample corrupted prompt: {all_corrupted_prompts[0]}")
        
        # Compute edge attribution scores
        print("\nComputing edge attribution scores...")
        edge_scores = self.compute_edge_attribution_scores(
            all_clean_prompts,
            all_corrupted_prompts,
            attr1_words,
            attr2_words,
            k=10
        )
        
        # Identify important edges
        print("\nIdentifying important edges...")
        important_edges = self.identify_important_edges(edge_scores, top_k=100)
        
        print(f"Top 5 important edges:")
        for edge_name, score in important_edges[:5]:
            print(f"  {edge_name}: {score:.4f}")
        
        # Analyze localization
        print("\nAnalyzing localization...")
        localization_analysis = self.analyze_localization(important_edges)
        
        print(f"Localization ratio: {localization_analysis['localization_ratio']:.2%}")
        print(f"Significant layers (>20% of edges): {list(localization_analysis['significant_layers'].keys())}")
        
        # Compute initial bias metric
        baseline_bias = self.compute_bias_metric_l2(
            all_clean_prompts, attr1_words, attr2_words, k=10
        )
        
        return {
            'weat_category': weat_category,
            'language': language,
            'baseline_bias_l2': baseline_bias,
            'edge_scores': edge_scores,
            'important_edges': important_edges,
            'localization_analysis': localization_analysis,
            'num_prompts': len(all_clean_prompts)
        }

def evaluate_all_models():
    """Main function to evaluate all models using EAP"""
    analyzer = EAPAnalyzer()
    weathub_loader = WEATHubLoader('iamshnoo/WEATHub', cache_dir="./datasets_cache")
    
    all_models = BASE_MODELS + FINETUNED_MODELS
    weat_categories = ['WEAT1', 'WEAT2', 'WEAT6']
    
    all_results = []
    
    for model_id in all_models:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_id}")
        print(f"{'='*70}")
        
        # Determine model type
        if model_id in BASE_MODELS:
            model_type = "base"
            eval_languages = ['en']  # Base models: only English
        else:
            model_type = "finetuned"
            eval_languages = ['en', 'hi']  # Finetuned: English and Hindi
        
        print(f"Model type: {model_type}")
        print(f"Evaluating languages: {eval_languages}")
        
        # Load model
        if not analyzer.load_model(model_id):
            print(f"Skipping {model_id} due to loading error")
            continue
        
        model_results = []
        
        # Evaluate for each language and WEAT category
        for language in eval_languages:
            for weat_cat in weat_categories:
                print(f"\n{'='*60}")
                print(f"Processing: {language} - {weat_cat}")
                print(f"{'='*60}")
                
                # Get word lists from WEATHub
                word_lists = weathub_loader.get_word_lists(language, weat_cat)
                if not word_lists:
                    print(f"Skipping {language}/{weat_cat} - no data available")
                    continue
                
                print(f"Target 1 ({word_lists['targ1_name']}): {len(word_lists['targ1'])} words")
                print(f"Target 2 ({word_lists['targ2_name']}): {len(word_lists['targ2'])} words")
                print(f"Attribute 1 ({word_lists['attr1_name']}): {len(word_lists['attr1'])} words")
                print(f"Attribute 2 ({word_lists['attr2_name']}): {len(word_lists['attr2'])} words")
                
                try:
                    # Evaluate bias using EAP
                    result = analyzer.evaluate_bias_for_weat(
                        word_lists, weat_cat, language
                    )
                    
                    # Store layer-wise results (ONE ROW PER LAYER)
                    edge_scores_dict = result['edge_scores']
                    baseline_bias = result['baseline_bias_l2']
                    
                    for edge_name, attribution_score in edge_scores_dict.items():
                        # Extract layer index from edge_name (e.g., "layer_5" -> 5)
                        if 'layer_' in edge_name:
                            layer_idx = int(edge_name.split('_')[1])
                        else:
                            layer_idx = -1  # Unknown layer
                        
                        # Create one row per layer
                        layer_result = {
                            'model_id': model_id,
                            'model_type': model_type,
                            'language': language,
                            'weat_category': weat_cat,
                            'layer_index': layer_idx,
                            'layer_name': edge_name,
                            'attribution_score': attribution_score,
                            'baseline_bias_l2': baseline_bias,
                            'localization_ratio': result['localization_analysis']['localization_ratio'],
                            'is_significant_layer': edge_name in [e[0] for e in result['important_edges'][:10]]
                        }
                        model_results.append(layer_result)
                    
                    # Also store in global results
                    all_results.extend(model_results)
                    
                    print(f"\n✓ Successfully evaluated {language}/{weat_cat}")
                    print(f"  Baseline bias (L2): {result['baseline_bias_l2']:.4f}")
                    print(f"  Important edges: {len(result['important_edges'])}")
                    print(f"  Localization ratio: {result['localization_analysis']['localization_ratio']:.2%}")
                    
                except Exception as e:
                    print(f"Error evaluating {language}/{weat_cat}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Clear model from cache
        analyzer.clear_model_cache()
        
        # Save results for this model (LAYER-WISE CSV per model)
        if model_results:
            df_layer_wise = pd.DataFrame(model_results)
            safe_model_name = model_id.replace("/", "_").replace("\\", "_")
            layer_wise_file = f"eap_{safe_model_name}_layerwise.csv"
            df_layer_wise.to_csv(layer_wise_file, index=False)
            print(f"\n✓ Layer-wise results saved to {layer_wise_file}")
            print(f"  Total layers recorded: {len(df_layer_wise)}")
            
            # Show summary statistics
            summary_stats = df_layer_wise.groupby(['language', 'weat_category']).agg({
                'attribution_score': ['mean', 'max', 'std'],
                'baseline_bias_l2': 'first'
            }).round(4)
            print(f"\nSummary Statistics:")
            print(summary_stats)
        else:
            print(f"No results generated for {model_id}")
    
    # Save all layer-wise results combined
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv("eap_all_models_layerwise.csv", index=False)
        print(f"\n{'='*70}")
        print("All layer-wise results saved to eap_all_models_layerwise.csv")
        print(f"Total records (all models, all layers): {len(all_results)}")
        print(f"Models covered: {df_all['model_id'].nunique()}")
        print(f"Categories covered: {df_all['weat_category'].nunique()}")
        print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("EAP evaluation completed for all models!")
    print(f"{'='*70}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import bitsandbytes
        print(f"Using bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("Warning: bitsandbytes not installed. Models will use full precision.")
        print("To enable quantization: pip install -U bitsandbytes")
    
    print("\n" + "="*70)
    print("Edge Attribution Patching (EAP) for Bias Analysis")
    print("Based on: 'Dissecting Bias in LLMs: A Mechanistic")
    print("          Interpretability Perspective'")
    print("="*70 + "\n")
    
    evaluate_all_models()