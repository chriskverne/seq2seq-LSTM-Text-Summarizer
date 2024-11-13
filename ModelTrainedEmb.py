import torch
import pandas as pd
from datasets import load_dataset
import re
from nltk.corpus import stopwords
import nltk
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
device_num = 0

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(device_num) if torch.cuda.is_available() else "No GPU")

ds = load_dataset("EdinburghNLP/xsum")

def clean_text(text):
  # Set to lower case
  text = text.lower()

  # Remove non-alphabetical
  text = re.sub(r'[^a-zA-Z\s]', '', text)

  # Remove additional whitespace
  text = ' '.join(text.split())
  # Remove stopwords (Let's keep them as stopwords are important for the english language)
  #words = text.split()
  #words = [word for word in words if word not in stop_words]
  #text = ' '.join(words)
  return text

ds = ds.map(lambda article: {
    'document' : clean_text(article['document']),
    'summary' : clean_text(article['summary'])
})

tokenizer = ByteLevelBPETokenizer(
    "./vocab.json",
    "./merges.txt"
)

tokenizer.add_special_tokens(["<pad>"])

# Get the pad token ID
pad_token_id = tokenizer.token_to_id('<pad>')

# Tokenize our inputs
def tokenize(paragraphs, doc_vec_size, sum_vec_size):
    # Tokenize the documents and summaries
    inputs = tokenizer.encode_batch(paragraphs['document'])
    summaries = tokenizer.encode_batch(paragraphs['summary'])

    # Convert to tensors and apply padding/truncation
    input_ids = [input.ids[:doc_vec_size] + [0] * (doc_vec_size - len(input.ids)) for input in inputs]
    label_ids = [summary.ids[:sum_vec_size] + [0] * (sum_vec_size - len(summary.ids)) for summary in summaries]

    # Convert to PyTorch tensors
    inputs = {'input_ids': torch.tensor(input_ids, dtype=torch.long)}
    inputs['labels'] = torch.tensor(label_ids, dtype=torch.long)

    return inputs

# Load embeddings for each token
def load_pretrained_embeddings(embedding_path, tokenizer, embedding_dim):
    # Load pretrained embeddings (e.g., GloVe) into a dictionary
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Create an embedding matrix where each row corresponds to a word in your tokenizer's vocabulary
    vocab_size = tokenizer.get_vocab_size()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for token, idx in tokenizer.get_vocab().items():
        embedding_vector = embeddings_index.get(token)
        if embedding_vector is not None:
            # Words found in the pretrained embeddings will have their corresponding vector
            embedding_matrix[idx] = embedding_vector
        else:
            # Words not found will be initialized randomly
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Model architecture
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pretrained_embeddings=None):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Optionally freeze the embeddings initially

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, input_ids):
        # Convert tokens to embeddings
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_len, embedding_dim]

        # Pass embeddings through LSTM
        outputs, (hidden, cell) = self.lstm(embedded)  # LSTM outputs: [batch_size, seq_len, hidden_dim]

        return outputs, (hidden, cell)
    
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pretrained_embeddings=None):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Load pretrained embeddings if available
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Optionally freeze embeddings initially

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output layer to predict token IDs

    def forward(self, input_ids, hidden, cell):
        # Embedding the input tokens
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_length, embedding_dim]

        # Passing through LSTM layers
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # Shape: [batch_size, seq_length, hidden_dim]

        # Output layer to predict the next token in the sequence
        predictions = self.fc(outputs)  # Shape: [batch_size, seq_length, vocab_size]

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tokenizer, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input_ids, target_ids, teacher_forcing_ratio=0.5):
        # Encode the input sequence
        encoder_outputs, (hidden, cell) = self.encoder(input_ids)

        # Prepare input and output sequences for the decoder
        batch_size = target_ids.size(0)
        target_length = target_ids.size(1)
        vocab_size = self.decoder.fc.out_features

        # Initialize outputs
        outputs = torch.zeros(batch_size, target_length, vocab_size).to(self.device)

        # Get the <s> (start token) token ID
        start_token_id = self.tokenizer.token_to_id('<s>')

        # First input to the decoder is the <s> token
        decoder_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long).to(self.device)

        # Decode for each time step
        for t in range(target_length):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = decoder_output.squeeze(1)

            # Teacher forcing: use the true target with some probability
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(2)  # Get predicted token
            decoder_input = target_ids[:, t].unsqueeze(1) if teacher_force else top1

        return outputs
  

def generate_summary(model, document, tokenizer, device, max_summary_length=100):
    model.eval()
    tokenized_doc = tokenizer.encode(document)
    
    input_ids = tokenized_doc.ids[:512]
    input_ids += [tokenizer.token_to_id('<pad>')] * (512 - len(input_ids))
    
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(input_ids)
        
        start_token_id = tokenizer.token_to_id("<s>")
        decoder_input = torch.full((1, 1), start_token_id, dtype=torch.long).to(device)
        
        summary_tokens = []
        eos_token_id = tokenizer.token_to_id("</s>")
        
        for _ in range(max_summary_length):
            decoder_output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            next_token = decoder_output.argmax(2)
            token_id = next_token.item()
            
            if token_id == eos_token_id:
                break
                
            summary_tokens.append(token_id)
            decoder_input = next_token
    
    return tokenizer.decode(summary_tokens)

class SummaryDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

class ModelCheckpoint:
    def __init__(self, save_dir, model_name="summarizer"):
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, model, optimizer, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"{self.model_name}_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model so far, save it separately
        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            
    def load(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, 
                checkpoint_handler, device, writer, teacher_forcing_ratio=0.5):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['labels'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, target_ids, teacher_forcing_ratio)
            
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            target_ids = target_ids.view(-1)
            
            loss = criterion(output, target_ids)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        checkpoint_handler.save(model, optimizer, epoch, val_loss, is_best)

def main():
    # Initialize tensorboard writer
    writer = SummaryWriter('runs/summarizer_experiment')
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    ds = load_dataset("EdinburghNLP/xsum")
    
    # Split training data into train and validation
    train_size = 200000
    val_size = 20000
    
    train_docs = [ds['train'][i]['document'] for i in range(train_size)]
    train_sums = [ds['train'][i]['summary'] for i in range(train_size)]
    
    val_docs = [ds['validation'][i]['document'] for i in range(train_size, train_size + val_size)]
    val_sums = [ds['validation'][i]['summary'] for i in range(train_size, train_size + val_size)]
    
    # Initialize tokenizer and create datasets
    tokenizer = ByteLevelBPETokenizer("./vocab.json", "./merges.txt")
    tokenizer.add_special_tokens(["<pad>"])
    
    train_tokens = tokenize({'document': train_docs, 'summary': train_sums}, 512, 100)
    val_tokens = tokenize({'document': val_docs, 'summary': val_sums}, 512, 100)
    
    train_dataset = SummaryDataset(train_tokens)
    val_dataset = SummaryDataset(val_tokens)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model components
    embedding_dim = 300
    hidden_dim = 1536
    num_layers = 5
    vocab_size = tokenizer.get_vocab_size()
    
    encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    decoder = DecoderLSTM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    model = Seq2Seq(encoder, decoder, tokenizer, device).to(device)
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<pad>'))
    
    # Initialize checkpoint handler
    checkpoint_handler = ModelCheckpoint('checkpoints')
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=30,
        checkpoint_handler=checkpoint_handler,
        device=device,
        writer=writer
    )
    
    # Test the model on some examples
    test_docs = [ds['test'][i]['document'] for i in range(10)]
    
    print("\nGenerating test summaries:")
    for i, doc in enumerate(test_docs):
        summary = generate_summary(model, doc, tokenizer, device)
        print(f"\nDocument {i + 1}:")
        print(f"Original: {doc[:200]}...")
        print(f"Summary: {summary}")

if __name__ == "__main__":
    main()




"""
# Model training
docs = [ds['train'][i]['document'] for i in range(200000)]
sums = [ds['train'][i]['summary'] for i in range(200000)]

# Tokenize the dataset
tokens = tokenize({'document': docs, 'summary': sums}, 512, 100)

# Create a SummaryDataset instance
dataset = SummaryDataset(tokens)

batch_size = 128
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model parameters
embedding_dim = 300
hidden_dim = 1536
num_layers = 5
vocab_size = tokenizer.get_vocab_size()
learning_rate = 0.001
num_epochs = 30
teacher_forcing_ratio = 0.5

pretrained_embedding_path = './glove.6B.300d.txt'
pretrained_embeddings = load_pretrained_embeddings(pretrained_embedding_path, tokenizer, embedding_dim)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
decoder = DecoderLSTM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
model = Seq2Seq(encoder, decoder, tokenizer, device).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

# Training function
def train(model, data_loader):
    model.train()
    epoch_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(input_ids, target_ids, teacher_forcing_ratio)

        # Reshape output and target for calculating loss
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        target_ids = target_ids.view(-1)

        # Calculate loss
        loss = criterion(output, target_ids)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

start_epoch = 0
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, data_loader)
    print(f"Training Loss: {train_loss:.4f}")
    #save_checkpoint(model, optimizer, epoch)

def generate_summary(model, document, max_summary_length=100):
    # Tokenize the document using encode, then pad and truncate manually
    model.eval()
    tokenized_doc = tokenizer.encode(document)
    
    # Truncate and pad manually to match your max_length setting
    input_ids = tokenized_doc.ids[:512]  # Truncate to max_length of 512
    input_ids += [pad_token_id] * (512 - len(input_ids))  # Pad to length of 512

    # Convert to PyTorch tensor and move to the correct device
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Encode the input document
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(input_ids)

    # Initialize decoder input with <s> token
    start_token_id = tokenizer.token_to_id("<s>")
    decoder_input = torch.full((1, 1), start_token_id, dtype=torch.long).to(device)
    summary_tokens = []

    # Greedily decode up to max_summary_length
    for _ in range(max_summary_length):
        with torch.no_grad():
            decoder_output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            next_token = decoder_output.argmax(2)
            summary_tokens.append(next_token.item())

            # Stop if we reach the end of the sequence
            if next_token.item() == tokenizer.token_to_id("</s>"):  # Replace with your end token
                break

            # Use the next token as the next input to the decoder
            decoder_input = next_token

    # Decode the generated tokens to text
    summary = tokenizer.decode(summary_tokens, skip_special_tokens=True)
    return summary


# Example documents for testing
test_docs = []

for i in range(200):
  test_docs.append(ds['train'][i]['document'])

# Generate and print summaries
for i, doc in enumerate(test_docs):
    summary = generate_summary(model, doc)
    print(f"Document {i + 1}: {doc}")
    print(f"Summary {i + 1}: {summary}")
    print("-" * 50)
"""