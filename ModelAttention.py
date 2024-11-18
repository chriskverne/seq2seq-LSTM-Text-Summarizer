import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import RobertaTokenizer
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import random
import os
import json
import re
from datetime import datetime

# Setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class Config:
    def __init__(self):
        self.batch_size = 16
        self.embedding_dim = 256
        self.hidden_dim = 512
        self.num_layers = 2
        self.dropout_rate = 0.3
        self.learning_rate = 0.0001
        self.num_epochs = 10
        self.beam_width = 5
        self.clip = 1.0
        self.teacher_forcing_ratio = 0.7
        self.doc_max_length = 512
        self.sum_max_length = 100
        self.save_dir = 'model_outputs'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(self.save_dir, exist_ok=True)

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
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs = self.layer_norm(outputs)
        
        # Handle bidirectional states
        hidden = self._reshape_hidden(hidden)
        cell = self._reshape_hidden(cell)
        return outputs, (hidden, cell)
    
    def _reshape_hidden(self, hidden):
        num_layers = hidden.shape[0] // 2
        hidden = torch.cat([hidden[2*i:2*i+2] for i in range(num_layers)], dim=2)
        return hidden

class ImprovedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Change the input dimension to match concatenated hidden states
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Reshape hidden to match encoder outputs dimensions
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attention(encoder_outputs))
        attention = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        
        # Calculate attention weights
        attention_weights = F.softmax(attention, dim=1)
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights

class EnhancedDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = ImprovedAttention(hidden_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,  # Input size includes embedding and context
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Changed from hidden_dim * 2
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, input_ids, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input_ids))
        
        # Get context using attention
        context, _ = self.attention(hidden[-1], encoder_outputs)
        
        # Combine embedding and context
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = self.layer_norm(output)
        
        # Make prediction using only the output
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
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = trg[:, t:t+1] if teacher_force else top1
            
        return outputs

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
                    candidates.append((new_sequence, new_score, 
                                    new_hidden, new_cell))
            
            beam = sorted(candidates, key=lambda x: x[1], 
                        reverse=True)[:beam_width]
            
            if len(finished_sequences) >= beam_width:
                break
        
        finished_sequences.extend(beam)
        best_sequence = max(finished_sequences, key=lambda x: x[1])[0]
        
        return best_sequence.squeeze().tolist()

class SummaryDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def prepare_data():
    ds = load_dataset("EdinburghNLP/xsum")
    
    #train_docs = [clean_text(item['document']) for item in ds['train']]
    #train_sums = [clean_text(item['summary']) for item in ds['train']]
    train_docs = [clean_text(ds['train'][i]['document']) for i in range(100)]
    train_sums = [clean_text(ds['train'][i]['summary']) for i in range(100)]
    
    #val_docs = [clean_text(item['document']) for item in ds['validation']]
    #val_sums = [clean_text(item['summary']) for item in ds['validation']]
    val_docs = [clean_text(ds['validation'][i]['document']) for i in range(100)]
    val_sums = [clean_text(ds['validation'][i]['summary']) for i in range(100)]
    
    train_encodings = tokenizer(
        train_docs,
        padding="max_length",
        truncation=True,
        max_length=config.doc_max_length,
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        train_labels = tokenizer(
            train_sums,
            padding="max_length",
            truncation=True,
            max_length=config.sum_max_length,
            return_tensors="pt"
        )
    
    val_encodings = tokenizer(
        val_docs,
        padding="max_length",
        truncation=True,
        max_length=config.doc_max_length,
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        val_labels = tokenizer(
            val_sums,
            padding="max_length",
            truncation=True,
            max_length=config.sum_max_length,
            return_tensors="pt"
        )
    
    train_encodings['labels'] = train_labels['input_ids']
    val_encodings['labels'] = val_labels['input_ids']
    
    return train_encodings, val_encodings

def train_model(model, train_loader, val_loader, optimizer, criterion, config):
    best_val_loss = float('inf')
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids, labels, config.teacher_forcing_ratio)
            
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                outputs = model(input_ids, labels, 0)  # No teacher forcing
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(config.save_dir, 'best_model.pt'))

def generate_summary(model, text, max_length=100, beam_width=5):
    model.eval()
    
    tokens = tokenizer(
        clean_text(text),
        return_tensors="pt",
        max_length=config.doc_max_length,
        padding="max_length",
        truncation=True
    )
    
    src = tokens["input_ids"].to(model.device)
    output_ids = beam_search(model, src, beam_width, max_length)
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return summary

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    print("Loading and preparing data...")
    train_encodings, val_encodings = prepare_data()
    
    # Create datasets and dataloaders
    train_dataset = SummaryDataset(train_encodings)
    val_dataset = SummaryDataset(val_encodings)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
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
    
    # Initialize optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, val_loader, optimizer, criterion, config)
    
    # Test the model
    print("\nTesting the model with a sample text...")
    sample_text = "The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town. First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit. However, she said more preventative work could have been carried out to ensure the retaining wall did not fail. It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten, she said. That may not be true but it is perhaps my perspective over the last few days. Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out? Meanwhile, a flood alert remains in place across the Borders because of the constant rain. Peebles was badly hit by problems, sparking calls to introduce more defences in the area. Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs. The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand. He said it was important to get the flood protection plan right but backed calls to speed up the process. I was quite taken aback by the amount of damage that has been done, he said. Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses. He said it was important that immediate steps were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans. Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk."
    
    summary = generate_summary(model, sample_text)
    print("\nOriginal text:", sample_text)
    print("\nGenerated summary:", summary)