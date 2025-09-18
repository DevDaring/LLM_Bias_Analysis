# -*- coding: utf-8 -*-
"""
SEAT (Sentence Encoder Association Test) for Layer-wise Bias Analysis
Evaluates bias in language models using sentence-level embeddings
Compatible with the existing WEAT-based research framework
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
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from huggingface_hub import login
import warnings
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
    "facebook/MobileLLM-125M": "meta-llama/Llama-2-7b-hf",  # Use Llama tokenizer for MobileLLM
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi": "meta-llama/Llama-2-7b-hf"
}

# Models requiring special tokenizer loading configuration
SPECIAL_TOKENIZER_CONFIG = {
    "facebook/MobileLLM-125M": {"use_fast": False},
    "DebK/MobileLLM-125M-finetuned-alpaca-hindi": {"use_fast": False},
    "cerebras/Cerebras-GPT-111M": {"use_fast": False},
    "DebK/cerebras-gpt-111m-finetuned-alpaca-hindi": {"use_fast": False}
}

# Logging utility function
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

class SEATEvaluator:
    """SEAT evaluation for transformer models"""
    
    def __init__(self, cache_dir="./hf_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Enhanced sentence templates for contextualization, specific to each category
        self.sentence_templates = {
            # Generic templates used as fallback
            'en': {
                'default': [
                    "This is {}.",
                    "{} is here.",
                    "I like {}.",
                    "{} is good.",
                    "The {} is nice.",
                    "{} makes me happy.",
                    "I saw {}.",
                    "{} is beautiful."
                ],
                # WEAT1: Flowers/Insects vs Pleasant/Unpleasant
                'WEAT1': [
                    "I saw a {} today.",
                    "The {} was in the garden.",
                    "There is a {} on the leaf.",
                    "A {} has interesting features.",
                    "Scientists study the {}.",
                    "Children are fascinated by {}.",
                    "The {} moved quickly.",
                    "I read about {} in a book."
                ],
                # WEAT2: Musical instruments/Weapons vs Pleasant/Unpleasant
                'WEAT2': [
                    "The {} was displayed in the museum.",
                    "Someone used the {} skillfully.",
                    "I learned how to use a {}.",
                    "The {} made a distinct sound.",
                    "They brought a {} to the event.",
                    "The {} belongs to my friend.",
                    "History shows many examples of {}.",
                    "The {} was carefully crafted."
                ],
                # WEAT6: Male/Female names vs Career/Family
                'WEAT6': [
                    "{} is a person I know.",
                    "I met {} yesterday.",
                    "{} lives in my neighborhood.",
                    "{} made an interesting choice.",
                    "{} has many responsibilities.",
                    "{} works hard every day.",
                    "{} has clear priorities.",
                    "{} is focused on their goals."
                ]
            },
            'hi': {
                'default': [
                    "यह {} है।",
                    "{} यहाँ है।",
                    "मुझे {} पसंद है।",
                    "{} अच्छा है।",
                    "{} सुंदर है।",
                    "{} मुझे खुश करता है।",
                    "मैंने {} देखा।",
                    "{} खूबसूरत है।"
                ],
                # WEAT1: Flowers/Insects vs Pleasant/Unpleasant
                'WEAT1': [
                    "मैंने आज एक {} देखा।",
                    "{} बगीचे में था।",
                    "पत्ती पर एक {} है।",
                    "{} की रोचक विशेषताएं हैं।",
                    "वैज्ञानिक {} का अध्ययन करते हैं।",
                    "बच्चे {} से आकर्षित होते हैं।",
                    "{} तेजी से चलता है।",
                    "मैंने एक किताब में {} के बारे में पढ़ा।"
                ],
                # WEAT2: Musical instruments/Weapons vs Pleasant/Unpleasant
                'WEAT2': [
                    "{} संग्रहालय में प्रदर्शित किया गया था।",
                    "किसी ने {} का कुशलता से उपयोग किया।",
                    "मैंने {} का उपयोग करना सीखा।",
                    "{} ने एक अलग आवाज निकाली।",
                    "वे कार्यक्रम में एक {} लाए।",
                    "{} मेरे दोस्त का है।",
                    "इतिहास {} के कई उदाहरण दिखाता है।",
                    "{} सावधानी से बनाया गया था।"
                ],
                # WEAT6: Male/Female names vs Career/Family
                'WEAT6': [
                    "{} एक व्यक्ति है जिसे मैं जानता हूं।",
                    "मैं कल {} से मिला।",
                    "{} मेरे पड़ोस में रहते हैं।",
                    "{} ने एक दिलचस्प विकल्प चुना।",
                    "{} की कई जिम्मेदारियां हैं।",
                    "{} हर दिन कड़ी मेहनत करते हैं।",
                    "{} की स्पष्ट प्राथमिकताएं हैं।",
                    "{} अपने लक्ष्यों पर केंद्रित हैं।"
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
        # Clear previous model
        self.clear_model_cache()
        
        tokenizer_id = TOKENIZER_MAPPING.get(model_id, model_id)
        
        print(f"Loading model: {model_id}")
        print(f"Loading tokenizer: {tokenizer_id}")
        
        # Determine if model needs trust_remote_code
        # Models that require custom code execution
        trust_remote = any(model_name in model_id for model_name in ["OpenELM", "MobileLLM"])
        log_debug(f"Setting trust_remote_code={trust_remote}")
        
        # Get special tokenizer config if available
        tokenizer_config = SPECIAL_TOKENIZER_CONFIG.get(model_id, {"use_fast": True})
        log_debug(f"Tokenizer config: {tokenizer_config}")
        
        # Try loading tokenizer with several fallback strategies
        try:
            # First attempt: Use configured tokenizer settings
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
            
            # Second attempt: Try opposite of use_fast setting
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
                log_debug(f"Tokenizer loaded with opposite fast setting: {type(self.tokenizer).__name__}")
                
            except Exception as e2:
                log_debug(f"Second tokenizer attempt failed: {str(e2)}")
                
                # Third attempt: Try with default Llama tokenizer as fallback
                try:
                    fallback_tokenizer = "meta-llama/Llama-2-7b-hf"
                    log_debug(f"Trying fallback tokenizer: {fallback_tokenizer}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        fallback_tokenizer,
                        cache_dir=self.cache_dir,
                        token=HF_TOKEN,
                        trust_remote_code=False  # Fallback tokenizer doesn't need custom code
                    )
                    log_debug(f"Using fallback tokenizer: {type(self.tokenizer).__name__}")
                    
                except Exception as e3:
                    print(f"Error: All tokenizer loading attempts failed.")
                    print(f"  First attempt: {str(e)}")
                    print(f"  Second attempt: {str(e2)}")
                    print(f"  Fallback attempt: {str(e3)}")
                    return False
        
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            log_debug("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Show tokenizer information
        log_debug(f"Vocabulary size: {len(self.tokenizer)}")
        log_debug(f"Tokenizer type: {type(self.tokenizer).__name__}")
        log_debug(f"Special tokens: {self.tokenizer.all_special_tokens}")
        
        # Configure quantization for memory efficiency
        try:
            log_debug("Setting up quantization config")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Check if bitsandbytes is properly installed
            try:
                import bitsandbytes as bnb
                log_debug(f"Bitsandbytes version: {bnb.__version__}")
            except ImportError:
                print("Warning: bitsandbytes not installed. Falling back to 8-bit or full precision.")
                quantization_config = None
                
            # Load model
            print(f"Loading model architecture: {model_id}")
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "cache_dir": self.cache_dir,
                "token": HF_TOKEN,
                "trust_remote_code": trust_remote
            }
            
            # Only add quantization config if it exists
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                
            log_debug(f"Model loading parameters: {model_kwargs}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            self.current_model_id = model_id
            log_debug(f"Model loaded successfully with {self.model.num_parameters()} parameters")
            print(f"Model {model_id} loaded successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model {model_id}: {error_msg}")
            
            # Specific handling for bitsandbytes errors
            if "bitsandbytes" in error_msg.lower():
                print("This appears to be a bitsandbytes-related error.")
                print("Try installing or updating bitsandbytes: pip install -U bitsandbytes")
            
            return False
        
        return True
    
    @torch.no_grad()
    def get_sentence_embedding(self, sentence: str, layer_idx: int):
        """Get sentence embedding from a specific layer"""
        try:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_idx]
            
            # Mean pooling over tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_state.size()).float()
            masked_embeddings = hidden_state * attention_mask
            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embedding = summed / summed_mask
            
            return embedding.mean(dim=0).float().cpu().numpy()
        except Exception as e:
            print(f"Error generating embedding for sentence '{sentence}' at layer {layer_idx}: {e}")
            # Return zero vector as fallback
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                return np.zeros(self.model.config.hidden_size)
            else:
                return np.zeros(768)  # Default size
    
    def compute_seat_score(self, word_lists, layer_idx, language):
        """Compute SEAT score for given word lists at a specific layer"""
        # Get category-specific templates if available, otherwise use default
        weat_category = word_lists.get('category', 'default')
        if weat_category in self.sentence_templates[language]:
            templates = self.sentence_templates[language][weat_category]
        else:
            templates = self.sentence_templates[language]['default']
            print(f"No specific templates for {weat_category}, using default")
        
        # Generate embeddings for all words with all templates
        def get_embeddings_for_words(words):
            embeddings = []
            for word in words:
                word_embeddings = []
                for template in templates:
                    sentence = template.format(word)
                    emb = self.get_sentence_embedding(sentence, layer_idx)
                    word_embeddings.append(emb)
                # Average across templates
                embeddings.append(np.mean(word_embeddings, axis=0))
            return np.array(embeddings)
        
        # Get embeddings for all groups
        T1_emb = get_embeddings_for_words(word_lists['targ1'])
        T2_emb = get_embeddings_for_words(word_lists['targ2'])
        A1_emb = get_embeddings_for_words(word_lists['attr1'])
        A2_emb = get_embeddings_for_words(word_lists['attr2'])
        
        # Compute SEAT score (similar to WEAT but with sentences)
        def s(w_emb, A_emb, B_emb):
            mean_cos_A = np.mean([cosine_similarity([w_emb], [a])[0][0] for a in A_emb])
            mean_cos_B = np.mean([cosine_similarity([w_emb], [b])[0][0] for b in B_emb])
            return mean_cos_A - mean_cos_B
        
        mean_T1 = np.mean([s(t, A1_emb, A2_emb) for t in T1_emb])
        mean_T2 = np.mean([s(t, A1_emb, A2_emb) for t in T2_emb])
        
        all_s = [s(t, A1_emb, A2_emb) for t in np.concatenate((T1_emb, T2_emb))]
        std_dev = np.std(all_s, ddof=1)
        
        if std_dev > 0:
            seat_score = (mean_T1 - mean_T2) / std_dev
        else:
            seat_score = 0
            
        return seat_score

def evaluate_all_models():
    """Main function to evaluate all models"""
    evaluator = SEATEvaluator()
    weathub_loader = WEATHubLoader('iamshnoo/WEATHub', cache_dir="./datasets_cache")
    
    all_models = BASE_MODELS + FINETUNED_MODELS
    weat_categories = ['WEAT1', 'WEAT2', 'WEAT6']
    languages = ['en', 'hi']
    
    for model_id in all_models:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_id}")
        print(f"{'='*50}")
        
        # Initialize results for this model
        model_results = []
        
        # Determine model type for comment
        if model_id in BASE_MODELS:
            model_type = "base"
        else:
            model_type = "finetuned"
        
        # Load model
        if not evaluator.load_model(model_id):
            print(f"Skipping {model_id} due to loading error")
            continue
        
        # Get number of layers - with better error handling
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
        
        # Determine which languages to evaluate based on model type
        if model_type == "base":
            # Base models: evaluate only English WEAT datasets
            eval_languages = ['en']
            print(f"Base model - evaluating only English WEAT datasets")
        else:
            # Finetuned models: evaluate both English and Hindi WEAT datasets
            eval_languages = ['en', 'hi']
            print(f"Finetuned model - evaluating both English and Hindi WEAT datasets")
        
        # Evaluate for each language and WEAT category
        for language in eval_languages:
            for weat_cat in weat_categories:
                print(f"\nProcessing: {language} - {weat_cat}")
                
                # Get word lists
                word_lists = weathub_loader.get_word_lists(language, weat_cat)
                if not word_lists:
                    continue
                
                # Add category info to word lists for template selection
                word_lists['category'] = weat_cat
                
                # Evaluate each layer
                for layer_idx in tqdm(range(num_layers), desc=f"Layers"):
                    try:
                        seat_score = evaluator.compute_seat_score(word_lists, layer_idx, language)
                        
                        # Create comment
                        comment = f"SEAT_{model_type}_{language}_{weat_cat}_layer{layer_idx}"
                        
                        model_results.append({
                            'model_id': model_id,
                            'language': language,
                            'weat_category_id': weat_cat,
                            'layer_idx': layer_idx,
                            'SEAT_score': seat_score,
                            'comments': comment
                        })
                    except Exception as e:
                        print(f"Error at layer {layer_idx}: {e}")
                        continue
        
        # Clear model from cache
        evaluator.clear_model_cache()
        
        # Save results for this model
        if model_results:
            df = pd.DataFrame(model_results)
            # Create safe filename from model_id
            safe_model_name = model_id.replace("/", "_").replace("\\", "_")
            output_file = f"seat_{safe_model_name}_results.csv"
            df.to_csv(output_file, index=False)
            print(f"\nResults for {model_id} saved to {output_file}")
            print(f"Generated {len(model_results)} result records")
            print(df.head())
        else:
            print(f"No results generated for {model_id}")
    
    print(f"\n{'='*60}")
    print("SEAT evaluation completed for all models!")
    print("Individual CSV files created for each model.")
    print("Run the merge script to combine all results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Check if bitsandbytes is available
    try:
        import bitsandbytes
        print(f"Using bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("Warning: bitsandbytes not installed. Models will be loaded in full precision.")
        print("To enable quantization, install bitsandbytes: pip install -U bitsandbytes")
    
    evaluate_all_models()