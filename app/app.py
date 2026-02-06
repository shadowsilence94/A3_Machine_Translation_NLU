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
scratch_model = None
sp_my = None
sp_en = None

def load_nllb():
    global nllb_model, nllb_tokenizer
    path = NLLB_PATH if os.path.exists(NLLB_PATH) else NLLB_PATH_SYNC if os.path.exists(NLLB_PATH_SYNC) else None
    
    if path:
        print(f"Loading NLLB from {path}...")
        try:
            nllb_tokenizer = AutoTokenizer.from_pretrained(path)
            nllb_model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
            print("NLLB Loaded.")
        except Exception as e:
            print(f"Failed to load NLLB: {e}")
    else:
        print("NLLB Model not found locally. Downloading fine-tuned model 'shadowsilence/burmese-nllb-model'...")
        try:
            # Use our fine-tuned model uploaded to HF Hub
            checkpoint = "shadowsilence/burmese-nllb-model"
            # NLLB requires src_lang be set for correct tokenizer behavior
            nllb_tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang="mya_Mymr", tgt_lang="eng_Latn")
            nllb_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(DEVICE)
            print("Remote Fine-Tuned NLLB Loaded.")
        except Exception as e:
            print(f"Failed to load Remote NLLB: {e}")
            print("Falling back to base model...")
            checkpoint = "facebook/nllb-200-distilled-600M"
            nllb_tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang="mya_Mymr", tgt_lang="eng_Latn")
            nllb_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(DEVICE)

def load_scratch_transformer():
    global scratch_model, sp_my, sp_en
    if os.path.exists(TRANSFORMER_PATH) and os.path.exists(SPM_MY_PATH) and os.path.exists(SPM_EN_PATH):
        print("Loading Scratch Transformer...")
        try:
            sp_my = spm.SentencePieceProcessor(model_file=SPM_MY_PATH)
            sp_en = spm.SentencePieceProcessor(model_file=SPM_EN_PATH)
            
            # Must match training config in notebook
            scratch_model = TransformerModel(len(sp_my), len(sp_en), d_model=256, nhead=4, 
                                            num_encoder_layers=2, num_decoder_layers=2, 
                                            dim_feedforward=512, dropout=0.1).to(DEVICE)
            
            scratch_model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
            scratch_model.eval()
            print("Scratch Transformer Loaded.")
        except Exception as e:
            print(f"Failed to load Scratch Transformer: {e}")
            scratch_model = None
    else:
        print("Scratch Transformer files not found.")

# Initial Load
load_nllb()
load_scratch_transformer()

# --- 3. Translation Logic ---
def translate_nllb(text):
    if not nllb_model or not nllb_tokenizer: return "Error: NLLB Model not loaded."
    nllb_tokenizer.src_lang = "mya_Mymr"
    inputs = nllb_tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=128)
    return nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def translate_scratch(text):
    if not scratch_model or not sp_my or not sp_en: return "Error: Scratch Model not available."
    
    # Encoded IDs list
    encoded_list = sp_my.encode_as_ids(text)
    # Add BOS and EOS tokens just like during training
    src_ids = [sp_my.bos_id()] + encoded_list + [sp_my.eos_id()]
    
    # Shape: [1, src_len] because batch_first=True
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(DEVICE)
    
    # Greedy Decode
    max_len = 50
    # Start with SOS (BOS)
    outputs = [sp_en.bos_id()]
    
    for i in range(max_len):
        # Shape: [1, curr_trg_len]
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = scratch_model(src_tensor, trg_tensor)
            # output shape: [1, curr_trg_len, vocab_size]
            # Get last token prediction
            best_guess = output.argmax(2)[:, -1].item()
        
        if best_guess == sp_en.eos_id(): 
             break
             
        outputs.append(best_guess)
        
    translated_text = sp_en.decode(outputs[1:]) # Skip start token
    return translated_text

# --- 4. Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    original = ""
    model_choice = "nllb"
    
    if request.method == 'POST':
        original = request.form['source_text']
        model_choice = request.form.get('model_choice', 'nllb')
        
        if original:
            if model_choice == 'nllb':
                translation = translate_nllb(original)
            else:
                translation = translate_scratch(original)
            
    return render_template('index.html', translation=translation, original=original, model_choice=model_choice)

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'nllb')
    
    if not text: return jsonify({'error': 'No text provided'}), 400
    
    if model_type == 'nllb':
        translation = translate_nllb(text)
    else:
        translation = translate_scratch(text)
        
    return jsonify({'translation': translation, 'model': model_type})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
