# -*- coding: utf-8 -*-
"""
ATLAS_Fixed_Models.py - Focused on 4 problematic models
Enhanced debugging and alternative extraction strategies for:
- EleutherAI/pythia-70m
- DebK/pythia-70m-finetuned-alpaca-hindi
- Qwen/Qwen2.5-1.5B
- DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full
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
from tqdm import tqdm
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# Set HF token
HF_TOKEN = "Secret"
login(HF_TOKEN, add_to_git_credential=True)

# ONLY the 4 problematic models
MODELS_TO_FIX = [
    "EleutherAI/pythia-70m",
    "DebK/pythia-70m-finetuned-alpaca-hindi",
    "Qwen/Qwen2.5-1.5B",
    "DebK/Qwen2.5-1.5B-finetuned-alpaca-hindi_full"
]

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
                    'target1': filtered[0]['targ1.examples'],
                    'target2': filtered[0]['targ2.examples'],
                    'attribute1': filtered[0]['attr1.examples'],
                    'attribute2': filtered[0]['attr2.examples']
                }
            else:
                print(f"Warning: No data found for language '{language_code}', category '{weat_category_id}'")
                return None
        except Exception as e:
            print(f"ERROR in get_word_lists: {e}")
            return None

class ATLASEvaluator:
    """Enhanced ATLAS evaluator with debugging for problematic models"""
    
    def __init__(self, model_id: str, device: str = 'cuda', attn_implementation: str = 'eager'):
        self.model_id = model_id
        self.device = device
        self.attn_implementation = attn_implementation
        
        print(f"\n{'='*80}")
        print(f"Initializing ATLAS for model: {model_id}")
        print(f"{'='*80}")
        
        # Load tokenizer
        print(f"Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"✗ Tokenizer loading failed: {e}")
            raise
        
        # Load model with multiple strategies
        print(f"Loading model with attn_implementation='{attn_implementation}'...")
        self.model = self._load_model_safe()
        self.model.eval()
        
        print(f"Model loaded on device: {self.device}")
        print(f"Model architecture: {self.model.config.model_type}")
        
    def _load_model_safe(self):
        """Load model with fallback strategies"""
        strategies = [
            {'attn': 'eager', 'dtype': torch.float32},  # Try FP32 first for Qwen
            {'attn': 'eager', 'dtype': torch.float16},
            {'attn': 'sdpa', 'dtype': torch.float16},
            {'attn': None, 'dtype': torch.float16},
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                print(f"  Attempt {i}: attn={strategy['attn']}, dtype={strategy['dtype']}")
                
                kwargs = {
                    'token': HF_TOKEN,
                    'torch_dtype': strategy['dtype'],
                    'device_map': 'auto',
                    'trust_remote_code': True,
                    'low_cpu_mem_usage': True
                }
                
                if strategy['attn']:
                    kwargs['attn_implementation'] = strategy['attn']
                
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
                print(f"  ✓ Model loaded successfully with strategy {i}")
                return model
                
            except Exception as e:
                print(f"  ✗ Strategy {i} failed: {str(e)[:100]}")
                if i == len(strategies):
                    raise
                continue
    
    def get_num_layers(self):
        """Get number of transformer layers with multiple detection methods"""
        config = self.model.config
        
        # Method 1: Standard attributes
        for attr in ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers']:
            if hasattr(config, attr):
                num_layers = getattr(config, attr)
                log_debug(f"Detected {num_layers} layers via config.{attr}")
                return num_layers
        
        # Method 2: Count actual layers in model
        try:
            if hasattr(self.model, 'transformer'):
                if hasattr(self.model.transformer, 'h'):
                    num_layers = len(self.model.transformer.h)
                    log_debug(f"Detected {num_layers} layers via model.transformer.h")
                    return num_layers
                elif hasattr(self.model.transformer, 'layers'):
                    num_layers = len(self.model.transformer.layers)
                    log_debug(f"Detected {num_layers} layers via model.transformer.layers")
                    return num_layers
            
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'layers'):
                    num_layers = len(self.model.model.layers)
                    log_debug(f"Detected {num_layers} layers via model.model.layers")
                    return num_layers
        except Exception as e:
            log_debug(f"Layer counting failed: {e}")
        
        # Fallback
        log_debug("WARNING: Could not detect number of layers, using default 12")
        return 12
    
    def get_embedding_layer(self):
        """Get embedding layer with multiple detection methods"""
        # Method 1: GPT-style (gpt_neox, gpt2)
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            log_debug("Using model.transformer.wte for embeddings")
            return self.model.transformer.wte
        
        # Method 2: GPT-NeoX style
        if hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'embed_in'):
            log_debug("Using model.gpt_neox.embed_in for embeddings")
            return self.model.gpt_neox.embed_in
        
        # Method 3: Llama/Qwen style
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            log_debug("Using model.model.embed_tokens for embeddings")
            return self.model.model.embed_tokens
        
        # Method 4: Direct embed_tokens
        if hasattr(self.model, 'embed_tokens'):
            log_debug("Using model.embed_tokens for embeddings")
            return self.model.embed_tokens
        
        log_debug("WARNING: Could not find embedding layer")
        return None
    
    def get_token_indices(self, text: str, word: str):
        """
        Find all token indices where 'word' appears in tokenized 'text'.
        Enhanced with multiple tokenization variations for robustness.
        """
        try:
            # Tokenize the full text
            tokens = self.tokenizer.tokenize(text)
            
            # Try multiple variations of the word
            word_variations = [
                word,
                ' ' + word,
                word + ' ',
                ' ' + word + ' ',
                word.lower(),
                ' ' + word.lower(),
                word.capitalize(),
                ' ' + word.capitalize()
            ]
            
            indices = []
            for variation in word_variations:
                word_tokens = self.tokenizer.tokenize(variation)
                if not word_tokens:
                    continue
                
                # Search for the word tokens in the full token list
                for i in range(len(tokens) - len(word_tokens) + 1):
                    if tokens[i:i+len(word_tokens)] == word_tokens:
                        indices.append(i)
                
                if indices:
                    log_debug(f"Found '{word}' at positions {indices} using variation '{variation}'", level=2)
                    return indices
            
            # Fallback: search for partial match
            for i, token in enumerate(tokens):
                if word.lower() in token.lower():
                    log_debug(f"Partial match: '{word}' found in token '{token}' at position {i}", level=2)
                    return [i]
            
            log_debug(f"WARNING: Could not find '{word}' in text", level=2)
            return []
            
        except Exception as e:
            log_debug(f"ERROR in get_token_indices: {e}", level=2)
            return []
    
    def extract_attention_scores(self, prompt: str, entity1: str, entity2: str, attribute: str):
        """
        ENHANCED: Extract attention scores AND probabilities with comprehensive fallbacks
        and verbose debugging for problematic models.
        """
        log_debug(f"Processing: E1='{entity1}', E2='{entity2}', Attr='{attribute}'", level=1)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        last_token_pos = inputs['attention_mask'].sum(dim=1) - 1
        log_debug(f"Sequence length: {inputs['input_ids'].shape[1]}, last_pos: {last_token_pos.item()}", level=2)
        
        # Get entity positions
        entity1_indices = self.get_token_indices(prompt, entity1)
        entity2_indices = self.get_token_indices(prompt, entity2)
        
        log_debug(f"Entity1 '{entity1}' indices: {entity1_indices}", level=2)
        log_debug(f"Entity2 '{entity2}' indices: {entity2_indices}", level=2)
        
        # Forward pass
        with torch.no_grad():
            try:
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    output_hidden_states=True
                )
                log_debug(f"Forward pass successful", level=2)
            except Exception as e:
                log_debug(f"ERROR in forward pass: {e}", level=2)
                return None
        
        # ====================
        # ENHANCED PROBABILITY EXTRACTION
        # ====================
        
        def extract_probability_robust(entity_name, entity_indices):
            """
            Multi-strategy probability extraction with extensive debugging
            """
            log_debug(f"Extracting probability for '{entity_name}'", level=3)
            
            # Strategy 1: Use entity tokens directly from tokenizer
            entity_tokens = self.tokenizer.encode(entity_name, add_special_tokens=False)
            log_debug(f"  Entity tokens: {entity_tokens}", level=3)
            
            if not entity_tokens:
                log_debug(f"  ✗ No tokens found for '{entity_name}'", level=3)
                return 0.0
            
            entity_token_id = entity_tokens[0]
            
            # Strategy 2: Check if we have valid logits
            if not hasattr(outputs, 'logits') or outputs.logits is None:
                log_debug(f"  ✗ No logits in model output", level=3)
                return 0.0
            
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            log_debug(f"  Logits shape: {logits.shape}", level=3)
            
            # Strategy 3a: Use position before entity (if entity found in prompt)
            if entity_indices and len(entity_indices) > 0:
                entity_pos = entity_indices[0]
                log_debug(f"  Entity position in sequence: {entity_pos}", level=3)
                
                if entity_pos > 0:
                    # Get logits at position BEFORE entity
                    prev_pos = entity_pos - 1
                    logits_at_prev = logits[prev_pos, :]
                    
                    # CRITICAL FIX: Use float32 for softmax to prevent NaN with FP16
                    logits_at_prev_f32 = logits_at_prev.float()
                    
                    # Check for inf/nan in logits
                    if torch.isinf(logits_at_prev_f32).any() or torch.isnan(logits_at_prev_f32).any():
                        log_debug(f"  ✗ Logits contain inf/nan at pos {prev_pos}", level=3)
                    else:
                        probs_at_prev = torch.softmax(logits_at_prev_f32, dim=-1)
                        
                        if entity_token_id < probs_at_prev.shape[0]:
                            prob_value = probs_at_prev[entity_token_id].item()
                            if not np.isnan(prob_value):
                                log_debug(f"  ✓ Prob at pos {prev_pos} (before entity): {prob_value:.6e}", level=3)
                                return prob_value
                            else:
                                log_debug(f"  ✗ Prob is NaN at pos {prev_pos}", level=3)
                        else:
                            log_debug(f"  ✗ Token ID {entity_token_id} out of vocab range", level=3)
            
            # Strategy 3b: Use last position in sequence
            last_pos = last_token_pos.item() if isinstance(last_token_pos, torch.Tensor) else last_token_pos
            log_debug(f"  Trying last position: {last_pos}", level=3)
            
            logits_at_last = logits[last_pos, :]
            logits_at_last_f32 = logits_at_last.float()
            
            if not (torch.isinf(logits_at_last_f32).any() or torch.isnan(logits_at_last_f32).any()):
                probs_at_last = torch.softmax(logits_at_last_f32, dim=-1)
                
                if entity_token_id < probs_at_last.shape[0]:
                    prob_value = probs_at_last[entity_token_id].item()
                    if not np.isnan(prob_value):
                        log_debug(f"  ✓ Prob at last pos {last_pos}: {prob_value:.6e}", level=3)
                        return prob_value
            
            # Strategy 3c: Average across all positions
            log_debug(f"  Trying average across all positions", level=3)
            logits_f32 = logits.float()
            
            if not (torch.isinf(logits_f32).any() or torch.isnan(logits_f32).any()):
                all_probs = torch.softmax(logits_f32, dim=-1)  # [seq_len, vocab_size]
                
                if entity_token_id < all_probs.shape[1]:
                    prob_value = all_probs[:, entity_token_id].mean().item()
                    if not np.isnan(prob_value):
                        log_debug(f"  ✓ Average prob across sequence: {prob_value:.6e}", level=3)
                        return prob_value
            
            # Strategy 3d: Max probability across sequence
            if entity_token_id < all_probs.shape[1]:
                prob_value = all_probs[:, entity_token_id].max().item()
                if not np.isnan(prob_value):
                    log_debug(f"  ✓ Max prob across sequence: {prob_value:.6e}", level=3)
                    return prob_value
            
            log_debug(f"  ✗ All probability extraction strategies failed", level=3)
            return 0.0
        
        prob_entity1 = extract_probability_robust(entity1, entity1_indices)
        prob_entity2 = extract_probability_robust(entity2, entity2_indices)
        
        log_debug(f"Final probs: E1={prob_entity1:.6e}, E2={prob_entity2:.6e}", level=2)
        
        # ====================
        # ATTENTION EXTRACTION (with fallbacks)
        # ====================
        
        attention_scores_e1 = []
        attention_scores_e2 = []
        num_layers = self.get_num_layers()
        
        # Check if attention is available
        use_attention = outputs.attentions is not None and len(outputs.attentions) > 0
        log_debug(f"Attention available: {use_attention}", level=2)
        
        if use_attention and entity1_indices and entity2_indices:
            # Primary method: Attention weights
            entity1_idx = entity1_indices[0]
            entity2_idx = entity2_indices[0]
            last_pos = last_token_pos.item() if isinstance(last_token_pos, torch.Tensor) else last_token_pos
            
            for layer_idx in range(min(len(outputs.attentions), num_layers)):
                attn_weights = outputs.attentions[layer_idx]
                
                if attn_weights is None:
                    log_debug(f"  Layer {layer_idx}: No attention weights", level=3)
                    use_attention = False
                    break
                
                attn_layer = attn_weights[0]  # [heads, seq_len, seq_len]
                
                # Average across attention heads
                attn_avg = attn_layer.mean(dim=0)  # [seq_len, seq_len]
                
                # Attention from last token to entities
                attn_e1 = attn_avg[last_pos, entity1_idx].item()
                attn_e2 = attn_avg[last_pos, entity2_idx].item()
                
                attention_scores_e1.append(attn_e1)
                attention_scores_e2.append(attn_e2)
            
            log_debug(f"Extracted attention for {len(attention_scores_e1)} layers", level=2)
        
        # FALLBACK 1: Hidden state similarity
        if not use_attention or len(attention_scores_e1) == 0:
            log_debug(f"Using FALLBACK: Hidden state similarity", level=2)
            
            if outputs.hidden_states and len(outputs.hidden_states) > 0:
                last_pos = last_token_pos.item() if isinstance(last_token_pos, torch.Tensor) else last_token_pos
                
                for layer_idx in range(min(len(outputs.hidden_states), num_layers)):
                    hidden = outputs.hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
                    
                    if entity1_indices and entity2_indices:
                        entity1_idx = entity1_indices[0]
                        entity2_idx = entity2_indices[0]
                        
                        # Cosine similarity between last token and entity tokens
                        last_hidden = hidden[last_pos]
                        entity1_hidden = hidden[entity1_idx]
                        entity2_hidden = hidden[entity2_idx]
                        
                        sim_e1 = torch.cosine_similarity(last_hidden.unsqueeze(0), entity1_hidden.unsqueeze(0)).item()
                        sim_e2 = torch.cosine_similarity(last_hidden.unsqueeze(0), entity2_hidden.unsqueeze(0)).item()
                        
                        # Convert similarity to attention-like score (normalize to [0, 1])
                        attn_e1 = (sim_e1 + 1) / 2
                        attn_e2 = (sim_e2 + 1) / 2
                        
                        attention_scores_e1.append(attn_e1)
                        attention_scores_e2.append(attn_e2)
                
                log_debug(f"Extracted {len(attention_scores_e1)} layers via hidden states", level=2)
        
        # FALLBACK 2: Embedding similarity
        if len(attention_scores_e1) == 0:
            log_debug(f"Using FALLBACK 2: Embedding similarity", level=2)
            
            embed_layer = self.get_embedding_layer()
            if embed_layer is not None:
                entity1_tokens = self.tokenizer.encode(entity1, add_special_tokens=False)
                entity2_tokens = self.tokenizer.encode(entity2, add_special_tokens=False)
                
                if entity1_tokens and entity2_tokens:
                    with torch.no_grad():
                        entity1_emb = embed_layer(torch.tensor([entity1_tokens[0]], device=self.device))
                        entity2_emb = embed_layer(torch.tensor([entity2_tokens[0]], device=self.device))
                        
                        # Get last token embedding from hidden states
                        if outputs.hidden_states and len(outputs.hidden_states) > 0:
                            last_pos = last_token_pos.item()
                            
                            for layer_idx in range(min(len(outputs.hidden_states), num_layers)):
                                last_hidden = outputs.hidden_states[layer_idx][0, last_pos, :]
                                
                                sim_e1 = torch.cosine_similarity(last_hidden.unsqueeze(0), entity1_emb).item()
                                sim_e2 = torch.cosine_similarity(last_hidden.unsqueeze(0), entity2_emb).item()
                                
                                attn_e1 = (sim_e1 + 1) / 2
                                attn_e2 = (sim_e2 + 1) / 2
                                
                                attention_scores_e1.append(attn_e1)
                                attention_scores_e2.append(attn_e2)
                            
                            log_debug(f"Extracted {len(attention_scores_e1)} layers via embeddings", level=2)
        
        # Verify we got results
        if len(attention_scores_e1) == 0:
            log_debug(f"✗✗✗ ALL FALLBACKS FAILED - No attention scores extracted", level=1)
            return None
        
        log_debug(f"✓ Extraction complete: {len(attention_scores_e1)} layers, probs=({prob_entity1:.6e}, {prob_entity2:.6e})", level=1)
        
        return {
            'attention_e1': attention_scores_e1,
            'attention_e2': attention_scores_e2,
            'prob_e1': prob_entity1,
            'prob_e2': prob_entity2
        }
    
    def evaluate_weat(self, language_code: str, weat_category_id: str, word_lists: dict, max_prompts: int = 10):
        """Evaluate WEAT category for given language"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {language_code.upper()} | {weat_category_id}")
        print(f"{'='*60}")
        
        target1 = word_lists['target1']
        target2 = word_lists['target2']
        attribute1 = word_lists['attribute1']
        attribute2 = word_lists['attribute2']
        
        results = []
        prompt_count = 0
        
        # Create prompts
        prompts = []
        for e1 in target1[:max_prompts]:
            for e2 in target2[:max_prompts]:
                for attr in attribute1[:max_prompts]:
                    prompts.append((e1, e2, attr))
                    prompt_count += 1
                    if prompt_count >= max_prompts:
                        break
                if prompt_count >= max_prompts:
                    break
            if prompt_count >= max_prompts:
                break
        
        print(f"Processing {len(prompts)} prompts...")
        
        for prompt_idx, (e1, e2, attr) in enumerate(tqdm(prompts, desc=f"{weat_category_id}")):
            prompt_text = f"Consider the words '{e1}' and '{e2}' in the context of '{attr}'."
            
            result = self.extract_attention_scores(prompt_text, e1, e2, attr)
            
            if result is None:
                log_debug(f"✗ Prompt {prompt_idx} failed completely", level=1)
                continue
            
            num_layers = len(result['attention_e1'])
            
            for layer_idx in range(num_layers):
                attn_e1 = result['attention_e1'][layer_idx]
                attn_e2 = result['attention_e2'][layer_idx]
                prob_e1 = result['prob_e1']
                prob_e2 = result['prob_e2']
                
                # Calculate bias ratio
                if prob_e2 > 1e-10:
                    bias_ratio = prob_e1 / prob_e2
                else:
                    bias_ratio = np.nan
                
                results.append({
                    'model_id': self.model_id,
                    'language': language_code,
                    'weat_category_id': weat_category_id,
                    'prompt_idx': prompt_idx,
                    'layer_idx': layer_idx,
                    'entity1': e1,
                    'entity2': e2,
                    'attribute': attr,
                    'attention_entity1': attn_e1,
                    'attention_entity2': attn_e2,
                    'prob_entity1': prob_e1,
                    'prob_entity2': prob_e2,
                    'bias_ratio': bias_ratio,
                    'comments': f"ATLAS_{'base' if not 'finetuned' in self.model_id else 'finetuned'}_{language_code}_{weat_category_id}_prompt{prompt_idx}_layer{layer_idx}"
                })
        
        print(f"✓ Completed: {len(results)} data points collected")
        
        # Calculate statistics
        valid_ratios = [r['bias_ratio'] for r in results if not np.isnan(r['bias_ratio'])]
        if valid_ratios:
            avg_bias = np.mean(valid_ratios)
            print(f"  Average bias ratio: {avg_bias:.4f}")
        else:
            print(f"  ✗✗✗ WARNING: No valid bias ratios computed!")
        
        return results

def main():
    """Main execution"""
    print("="*80)
    print("ATLAS_Fixed_Models.py - Focused Evaluation for 4 Problematic Models")
    print("="*80)
    print(f"Models to fix: {len(MODELS_TO_FIX)}")
    for model in MODELS_TO_FIX:
        print(f"  - {model}")
    print()
    
    # Load WEATHub
    weathub = WEATHubLoader("iamshnoo/wea_thub", cache_dir="./datasets_cache")
    
    # WEAT categories
    weat_categories = ['WEAT1', 'WEAT6', 'WEAT7']
    
    # Process each model
    for model_id in MODELS_TO_FIX:
        print(f"\n{'#'*80}")
        print(f"# Processing: {model_id}")
        print(f"{'#'*80}")
        
        try:
            # Determine languages
            if 'finetuned' in model_id.lower() and 'hindi' in model_id.lower():
                languages = ['en', 'hi']
            else:
                languages = ['en']
            
            print(f"Languages: {languages}")
            
            # Initialize evaluator
            evaluator = ATLASEvaluator(model_id, device='cuda', attn_implementation='eager')
            
            all_results = []
            
            # Process each language
            for lang in languages:
                print(f"\n{'-'*60}")
                print(f"Language: {lang.upper()}")
                print(f"{'-'*60}")
                
                for weat_cat in weat_categories:
                    word_lists = weathub.get_word_lists(lang, weat_cat)
                    
                    if word_lists:
                        results = evaluator.evaluate_weat(lang, weat_cat, word_lists, max_prompts=10)
                        all_results.extend(results)
                    else:
                        print(f"✗ No word lists for {lang}/{weat_cat}")
            
            # Save results
            if all_results:
                df = pd.DataFrame(all_results)
                
                # Create safe filename
                safe_model_name = model_id.replace('/', '_')
                output_file = f"atlas_{safe_model_name}_results.csv"
                
                df.to_csv(output_file, index=False)
                print(f"\n✓✓✓ Results saved to: {output_file}")
                print(f"    Total data points: {len(df)}")
                
                # Quick stats
                valid_count = df['bias_ratio'].notna().sum()
                print(f"    Valid bias ratios: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)")
            else:
                print(f"\n✗✗✗ No results collected for {model_id}")
            
            # Cleanup
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n✗✗✗ ERROR processing {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("ATLAS evaluation completed!")
    print("="*80)

if __name__ == "__main__":
    main()
