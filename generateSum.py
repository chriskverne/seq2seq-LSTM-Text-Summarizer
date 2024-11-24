import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re
import os
from datetime import datetime

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class Config:
    def __init__(self):
        self.batch_size = 96
        self.embedding_dim = 384
        self.hidden_dim = 512
        self.num_layers = 4
        self.dropout_rate = 0.2
        self.doc_max_length = 768 
        self.sum_max_length = 128
        self.beam_width = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Copy all the model classes exactly as they are
class ImprovedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attention(encoder_outputs))
        attention = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights

class EnhancedLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.num_layers = num_layers
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs = self.layer_norm(outputs)
        hidden = hidden.view(self.num_layers, 2, -1, hidden.size(2))
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = cell.view(self.num_layers, 2, -1, cell.size(2))
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        return outputs, (hidden, cell)

class EnhancedDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = ImprovedAttention(hidden_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, input_ids, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input_ids))
        context, _ = self.attention(hidden[-1], encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = self.layer_norm(output)
        prediction = self.fc(output)
        return prediction, hidden, cell

class ImprovedSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = torch.full((batch_size, 1), tokenizer.cls_token_id, 
                                 dtype=torch.long).to(self.device)
        
        for t in range(trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t:t+1] = output
            top1 = output.argmax(2)
            decoder_input = top1
            
        return outputs

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def beam_search(model, src, beam_width=5, max_length=100):
    model.eval()
    
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src)
        beam = [(torch.full((1, 1), tokenizer.cls_token_id,
                          dtype=torch.long).to(model.device), 0.0, hidden, cell)]
        finished_sequences = []
        
        for _ in range(max_length):
            candidates = []
            
            for sequence, score, hidden, cell in beam:
                if sequence[0, -1].item() == tokenizer.sep_token_id:
                    finished_sequences.append((sequence, score))
                    continue
                    
                output, new_hidden, new_cell = model.decoder(
                    sequence[:, -1:], hidden, cell, encoder_outputs)
                
                log_probs, indices = output[0, -1].topk(beam_width)
                
                for log_prob, idx in zip(log_probs, indices):
                    new_sequence = torch.cat([sequence, 
                                           idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    candidates.append((new_sequence, new_score, new_hidden, new_cell))
            
            if candidates:
                beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if len(finished_sequences) >= beam_width:
                break
        
        finished_sequences.extend(beam)
        best_sequence = max(finished_sequences, key=lambda x: x[1])[0] if finished_sequences else beam[0][0]
        return best_sequence.squeeze().tolist()

def generate_summary(model, text, max_length=100, beam_width=5):
    model.eval()
    tokens = tokenizer(
        clean_text(text),
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    src = tokens["input_ids"].to(model.device)
    output_ids = beam_search(model, src, beam_width, max_length)
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary

def main():
    # Initialize configuration
    config = Config()
    print(f"Using device: {config.device}")

    # Initialize model components
    encoder = EnhancedLSTMEncoder(
        tokenizer.vocab_size,
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout_rate
    ).to(config.device)
    
    decoder = EnhancedDecoder(
        tokenizer.vocab_size,
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout_rate
    ).to(config.device)
    
    model = ImprovedSeq2Seq(encoder, decoder, config.device).to(config.device)
    
    # Load checkpoint
    checkpoint_path = './model_checkpoints3/best_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with train loss: {checkpoint['train_loss']:.4f}")
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("EdinburghNLP/xsum")
    
    # Create output directory
    os.makedirs('./generated_summaries', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'./generated_summaries/summaries.txt'
    
    print(f"Generating 1000 summaries... Output will be saved to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(5000)):
            try:
                # Get article and real summary
                article = ds['train'][i]['document']
                real_summary = ds['train'][i]['summary']
                
                # Generate summary
                generated_summary = generate_summary(
                    model,
                    article,
                    max_length=config.sum_max_length,
                    beam_width=config.beam_width
                )
                
                # Write to file
                f.write("REAL SUMMARY:\n")
                f.write(f"{real_summary}\n\n")
                f.write("GENERATED SUMMARY:\n")
                f.write(f"{generated_summary}\n")
                
                # Print progress update
                if (i + 1) % 10 == 0:
                    print(f"\nCompleted {i + 1} summaries")
                    print(f"Latest summary length: {len(generated_summary.split())}")
                
            except Exception as e:
                print(f"\nError processing example {i+1}: {str(e)}")
                continue

if __name__ == "__main__":
    main()