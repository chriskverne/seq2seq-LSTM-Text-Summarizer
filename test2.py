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
import re
import shutil
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup  # Add this import at the top

# Setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class Config:
    def __init__(self):
        self.batch_size = 128
        self.embedding_dim = 512
        self.hidden_dim = 768
        self.num_layers = 4
        self.dropout_rate = 0.1
        self.learning_rate = 0.0001
        self.num_epochs = 35
        self.beam_width = 5
        self.clip = 1.0
        self.teacher_forcing_ratio = 0.7
        self.doc_max_length = 512
        self.sum_max_length = 128
        self.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        self.weight_decay = 0.01
        self.label_smoothing = 0.1
        self.gradient_accumulation_steps = 2

class EnhancedLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Keep this as hidden_dim // 2 for bidirectional
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
        
        # Properly reshape hidden and cell states for the decoder
        # For each layer, concatenate forward and backward states
        hidden = hidden.view(self.num_layers, 2, -1, hidden.size(2))  # [num_layers, 2, batch, hidden_dim//2]
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)  # [num_layers, batch, hidden_dim]
        
        cell = cell.view(self.num_layers, 2, -1, cell.size(2))
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        return outputs, (hidden, cell)

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
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = trg[:, t:t+1] if teacher_force else top1
            
        return outputs

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
            
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if len(finished_sequences) >= beam_width:
                break
        
        finished_sequences.extend(beam)
        best_sequence = max(finished_sequences, key=lambda x: x[1])[0]
        return best_sequence.squeeze().tolist()

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

def save_checkpoint(model, optimizer, epoch, train_loss, is_best):
    """
    Save model checkpoint, keeping only the best model based on training loss.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
    }
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('./model_checkpoints2', exist_ok=True)
    
    if is_best:
        # Remove previous best checkpoint if it exists
        checkpoint_path = './model_checkpoints2/best_checkpoint.pt'
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        # Save new best checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved new best checkpoint with training loss: {train_loss:.4f}")

def generate_test_summaries(model, config):
    """
    Generate and save 1000 test summaries with their original documents and real summaries.
    """
    model.eval()
    ds = load_dataset("EdinburghNLP/xsum")
    
    # Create output directory if it doesn't exist
    os.makedirs('./model_outputs', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'./model_outputs/test_summaries_{timestamp}.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(1000), desc="Generating test summaries"):
            article = ds['test'][i]['document']
            real_summary = ds['test'][i]['summary']
            
            # Generate summary
            generated_summary = generate_summary(
                model,
                article,
                max_length=config.sum_max_length,
                beam_width=config.beam_width
            )
            
            # Write to file
            f.write(f"Example #{i+1}\n")
            f.write("="* 80 + "\n\n")
            f.write("ORIGINAL DOCUMENT:\n")
            f.write(article + "\n\n")
            f.write("REAL SUMMARY:\n")
            f.write(real_summary + "\n\n")
            f.write("GENERATED SUMMARY:\n")
            f.write(generated_summary + "\n\n")
            f.write("-"* 80 + "\n\n")
    
    print(f"Test summaries saved to: {output_file}")
    return output_file

def prepare_data():
    ds = load_dataset("EdinburghNLP/xsum")
    
    # Use small subset for testing - change slice size for full training
    train_docs = [clean_text(ds['train'][i]['document']) for i in range(len(ds['train']))]
    train_sums = [clean_text(ds['train'][i]['summary']) for i in range(len(ds['train']))]
    val_docs = [clean_text(ds['validation'][i]['document']) for i in range(len(ds['validation']))]
    val_sums = [clean_text(ds['validation'][i]['summary']) for i in range(len(ds['validation']))]
    
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
    best_train_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Add gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Add warmup scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = num_training_steps // 10  # 10% of total steps for warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        
        # Add progress tracking
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Add mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels, config.teacher_forcing_ratio)
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                loss = criterion(outputs, labels)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'train_loss': f"{loss.item():.4f}"})
            
            # Update learning rate with warmup scheduler
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                outputs = model(input_ids, labels, 0)
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save checkpoint if training loss improved
        is_best = avg_train_loss < best_train_loss
        if is_best:
            best_train_loss = avg_train_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                is_best=True
            )
        
        # Log progress
        print(f"\nEpoch {epoch+1}")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        print(f"Best Train Loss: {best_train_loss:.4f}")
        
        # Early stopping
        if epoch > 5 and all(train_losses[-1] > loss for loss in train_losses[-6:-1]):
            print("\nTraining loss hasn't improved for 5 epochs.")
            #break
    
    return train_losses, val_losses, best_train_loss

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Check if we want to train or evaluate
    train_model_flag = True  # Set to False to only evaluate
    
    if train_model_flag:
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
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, optimizer, criterion, config
        )
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    else:
        # Load the best model for evaluation
        print("Loading best model for evaluation...")
        checkpoint_path = './model_checkpoints2/best_checkpoint.pt'
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No saved model found! Please train the model first.")
        
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
        
        # Load the saved model state
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate test summaries
        print("\nGenerating test summaries...")
        output_file = generate_test_summaries(model, config)
        print(f"\nTest summaries have been saved to: {output_file}")