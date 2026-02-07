import os
import torch
import torch.nn as nn
import sentencepiece as spm
import math
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Transformer from Scratch Definition ---
# --- 1. Transformer from Scratch Definition ---
class TransformationModel(nn.Module):
    # NOTE: Class name in notebook might have been TransformerModel, but let's check if user renamed it
    # The user's notebook has 'TransformerModel'.
    pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
                 d_model=512, nhead=8, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1, pad_idx=0):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        
        # Output Layer
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        # Create masks
        src_key_padding_mask = (src == self.pad_idx)
        # trg_key_padding_mask = (trg == self.pad_idx) # Optional, usually handled by generating loop mask
        
        # Target mask for autoregressive decoding
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(src.device)
        
        # Embed + Positional Encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        trg_emb = self.trg_embedding(trg) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoder(src_emb)
        trg_emb = self.pos_encoder(trg_emb)
        
        # Transformer Forward
        output = self.transformer(
            src=src_emb, 
            tgt=trg_emb, 
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            # tgt_key_padding_mask=trg_key_padding_mask
        )
        
        return self.fc_out(output)

# --- 2. Load Models ---
# Paths
BASE_DIR = os.path.dirname(__file__)
NLLB_PATH = os.path.join(BASE_DIR, 'nllb_model')
NLLB_PATH_SYNC = os.path.join(BASE_DIR, '../../nllb_model')
TRANSFORMER_PATH = os.path.join(BASE_DIR, 'models/transformer_model.pt')
SPM_MY_PATH = os.path.join(BASE_DIR, 'models/spm_my.model')
SPM_EN_PATH = os.path.join(BASE_DIR, 'models/spm_en.model')

# Global Variables
nllb_model = None
nllb_tokenizer = None
# Global Variables for Scratch Models
scratch_models = {}
sp_src_models = {}
sp_trg_models = {}

# Language Mapping for NLLB
NLLB_LANG_MAP = {
    'my': 'mya_Mymr',
    'th': 'tha_Thai',
    'zh': 'zho_Hans',
    'hi': 'hin_Deva',
    'ne': 'npi_Deva',
    'ur': 'urd_Arab',
    'vi': 'vie_Latn',
    'tl': 'tgl_Latn',
    'kk': 'kaz_Cyrl',
    'bn': 'ben_Beng',
    'de': 'deu_Latn'
}

def load_nllb():
    global nllb_model, nllb_tokenizer
    try:
        print("Loading NLLB Model...")
        # Check if model exists locally
        if os.path.exists(NLLB_PATH) or os.path.exists(NLLB_PATH_SYNC):
             model_path = NLLB_PATH if os.path.exists(NLLB_PATH) else NLLB_PATH_SYNC
             print(f"Loading from {model_path}...")
             nllb_tokenizer = AutoTokenizer.from_pretrained(model_path)
             nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
        else:
             # Download if not found (fallback)
             print("NLLB model not found locally. Downloading facebook/nllb-200-distilled-600M...")
             nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
             nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(DEVICE)
             
             # Save for later
             print(f"Saving NLLB model to {NLLB_PATH}...")
             nllb_tokenizer.save_pretrained(NLLB_PATH)
             nllb_model.save_pretrained(NLLB_PATH)
             
        print("NLLB Model Loaded.")
    except Exception as e:
        print(f"Failed to load NLLB Model: {e}")

def translate_nllb(text, src_lang="mya_Mymr", tgt_lang="eng_Latn"):
    if not nllb_model or not nllb_tokenizer: return "Error: NLLB Model not loaded. Please wait for the model to download or check logs."
    try:
        # Set source language
        nllb_tokenizer.src_lang = src_lang
        
        inputs = nllb_tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(tgt_lang), max_length=128)
        return nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error during NLLB translation: {e}")
        return f"Error translating: {str(e)}"

# Initial Load
load_nllb()

def load_scratch_transformer():
    global scratch_models, sp_src_models, sp_trg_models
    
    languages = ['my', 'th', 'zh', 'hi', 'ne', 'ur', 'vi', 'tl', 'kk', 'bn', 'de']
    
    for lang in languages:
        # Define paths for each language
        t_name = f'transformer_model_{lang}.pt' if lang != 'my' else 'transformer_model.pt'
        s_name = f'spm_{lang}.model'
        # English tokenizer naming convention
        if lang == 'my': e_name = 'spm_en.model'
        elif lang in ['th', 'zh', 'hi', 'ne', 'ur', 'vi', 'tl', 'kk', 'bn', 'de']: e_name = f'spm_en_{lang}.model'
        else: e_name = 'spm_en.model'
        
        # Check local then sync
        t_path = os.path.join(BASE_DIR, f'models/{t_name}')
        if not os.path.exists(t_path): t_path = os.path.join(BASE_DIR, f'../../models/{t_name}') # Fallback logic if needed, but standard is models/
        
        s_path = os.path.join(BASE_DIR, f'models/{s_name}')
        e_path = os.path.join(BASE_DIR, f'models/{e_name}')
        
        # Fix for standard deployment structure (app/models) vs dev
        if not os.path.exists(t_path):
             # Try sync path logic for dev
             t_path = os.path.join(BASE_DIR, f'../../app/models/{t_name}')
             s_path = os.path.join(BASE_DIR, f'../../app/models/{s_name}')
             e_path = os.path.join(BASE_DIR, f'../../app/models/{e_name}')

        if os.path.exists(t_path) and os.path.exists(s_path) and os.path.exists(e_path):
            try:
                print(f"Loading Scratch Model for {lang}...")
                sp_src_models[lang] = spm.SentencePieceProcessor(model_file=s_path)
                sp_trg_models[lang] = spm.SentencePieceProcessor(model_file=e_path)
                
                # Model params must match notebooks
                # New languages use vocab_size=8000
                vocab_size = 8000 if lang in ['hi', 'ne', 'ur', 'vi', 'tl', 'kk', 'bn', 'de'] else 4000
                
                model = TransformerModel(
                    src_vocab_size=vocab_size, 
                    trg_vocab_size=vocab_size, 
                    d_model=256, nhead=4, num_encoder_layers=2, 
                    num_decoder_layers=2, dim_feedforward=512, dropout=0.1, pad_idx=0
                ).to(DEVICE)
                
                model.load_state_dict(torch.load(t_path, map_location=DEVICE))
                model.eval()
                scratch_models[lang] = model
                print(f"Scratch Transformer ({lang}) Loaded.")
            except Exception as e:
                print(f"Failed to load Scratch Transformer ({lang}): {e}")
        else:
            print(f"Scratch Transformer files for {lang} not found. Skipping.")

def translate_scratch(text, lang='my'):
    # Lazy loading if model not found
    if lang not in scratch_models:
        print(f"Model for {lang} not found. Attempting to load...")
        load_scratch_transformer()
        
    if lang not in scratch_models:
        return f"Error: Model for {lang} not available. Please train it first."
    
    model = scratch_models[lang]
    sp_src = sp_src_models[lang]
    sp_trg = sp_trg_models[lang]
    
    encoded_list = sp_src.encode_as_ids(text)
    src_ids = [sp_src.bos_id()] + encoded_list + [sp_src.eos_id()]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(DEVICE)
    
    outputs = [sp_trg.bos_id()]
    for i in range(50):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
            best_guess = output.argmax(2)[:, -1].item()
        if best_guess == sp_trg.eos_id(): break
        outputs.append(best_guess)
        
    return sp_trg.decode(outputs[1:])

# --- 4. Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    original = ""
    model_choice = "nllb" # This will now effectively allow NLLB vs Scratch
    lang_choice = "my"
    
    if request.method == 'POST':
        original = request.form.get('source_text', '')
        model_choice = request.form.get('model_choice', 'nllb')
        lang_choice = request.form.get('lang_choice', 'my')
        
        if original:
            if model_choice == 'nllb':
                # Use NLLB with language code
                src_code = NLLB_LANG_MAP.get(lang_choice, 'mya_Mymr')
                translation = translate_nllb(original, src_lang=src_code, tgt_lang='eng_Latn')
            else:
                translation = translate_scratch(original, lang=lang_choice)
            
    return render_template('index.html', translation=translation, original=original, model_choice=model_choice, lang_choice=lang_choice)

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'nllb')
    lang = data.get('lang', 'my')
    direction = data.get('direction', 'f2e') # f2e (Foreign to English) or e2f (English to Foreign)
    
    if not text: return jsonify({'error': 'No text provided'}), 400
    
    # Language Mapping for NLLB
    # Language Mapping for NLLB (Use Global)
    target_code = NLLB_LANG_MAP.get(lang, 'mya_Mymr')
    english_code = 'eng_Latn'
    
    if model_type == 'nllb':
        if direction == 'f2e':
            # Foreign -> English
            translation = translate_nllb(text, src_lang=target_code, tgt_lang=english_code)
        else:
            # English -> Foreign
            translation = translate_nllb(text, src_lang=english_code, tgt_lang=target_code)
    else:
        # Scratch model
        if direction == 'e2f':
             translation = f"Error: The Scratch Transformer model only supports {lang.upper()} -> English translation. Please use NLLB for English -> {lang.upper()}."
        else:
             translation = translate_scratch(text, lang=lang)
        
    return jsonify({'translation': translation, 'model': model_type, 'lang': lang, 'direction': direction})

# Load Scratch Models
load_scratch_transformer()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
