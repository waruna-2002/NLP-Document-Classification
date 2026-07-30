import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# train_rf.py එකෙන් dataset එක load කරන function එක import කිරීම
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from train_rf import load_dataset_from_filepaths
from text_utils import clean_text

# --- 1. PyTorch Dataset Structure ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = str(self.texts[idx]).split()
        # Convert tokens to numerical IDs using vocabulary
        ids = [self.vocab.get(token, self.vocab.get("<UNK>", 1)) for token in tokens]
        
        # Padding / Truncating to fixed length
        if len(ids) < self.max_len:
            ids = ids + [self.vocab.get("<PAD>", 0)] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# --- 2. BiLSTM Model Architecture ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2, n_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate forward and backward final hidden states
        hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_last))

# --- 3. Main Training Pipeline ---
def train_bilstm():
    print("Loading Dataset for BiLSTM...")
    df = load_dataset_from_filepaths()

    if df.empty:
        print("Error: No data loaded! Please check dataset folder paths.")
        return

    print("Cleaning text data...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Label Mapping
    labels_map = {"COMPANY_SENSITIVE": 0, "PERSONAL": 1}
    df['label_id'] = df['label'].map(labels_map)

    # Build Vocabulary
    print("Building Vocabulary...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in df['clean_text']:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    X = df['clean_text'].values
    y = df['label_id'].values

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = TextDataset(X_train, y_train, vocab)
    test_dataset = TextDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMClassifier(vocab_size=len(vocab), output_dim=len(labels_map)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("\nTraining BiLSTM Neural Network...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- BiLSTM Classification Report ---")
    target_names = [k for k, v in sorted(labels_map.items(), key=lambda item: item[1])]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Save Model & Vocabulary to models/ directory
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    os.makedirs(models_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(models_dir, "bilstm_model.pth"))
    joblib.dump(vocab, os.path.join(models_dir, "vocab.pkl"))
    joblib.dump(labels_map, os.path.join(models_dir, "labels_map.pkl"))

    print(f"BiLSTM model, vocab, and labels map successfully saved to: {models_dir}")

if __name__ == "__main__":
    train_bilstm()