import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from google.colab import drive # type: ignore
from collections import Counter
from transformers import get_linear_schedule_with_warmup
import os

hyperparameters = {
    "lr": 1e-4,
    "batch_size": 1024,
    "embed_dim": 256,
    "epochs": 100,
    "num_warmup_steps": 500,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "num_heads": 4,
    "num_transformer_layers": 4,
    "ff_dim": 512,
    "cnn_out_channels": 64,
    "k-mers": 3,
    "max_len": 199
}

info = {
    "dataset_size": "data",
    "precision": "FP16",
    "dir_name": "Binary Mutation Model",
    "run": "5thRun",
    "optimizer": "Adam",
    "is_pre_training": False
}

# To mount large files from drive
drive.mount('/content/drive')

# Load large files from drive
data = f"/content/drive/MyDrive/{info['dataset_size']}.csv"
datao = pd.read_csv(data)

# Feature Generation

def gc_content(seq):
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq)

def at_content(seq):
    seq = seq.upper()
    return (seq.count('A') + seq.count('T')) / len(seq)

def is_cpg_site(row):
    seq = row['sequence'].upper()
    pos = row['mutation_pos']

    if pos < len(seq)-1 and seq[pos] == 'C' and seq[pos+1] == 'G':
        return 1

    if pos > 0 and seq[pos-1] == 'C' and seq[pos] == 'G':
        return 1
    return 0

def sequence_entropy(seq):
    counts = Counter(seq)
    total = len(seq)
    probs = [count / total for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def is_transition(ref, alt):
    transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
    return 1 if (ref, alt) in transitions else 0

chrom_lengths = {
    'chr1': 248956422,
    'chr2': 242193529,
    'chr3': 198295559,
    'chr4': 190214555,
    'chr5': 181538259,
    'chr6': 170805979,
    'chr7': 159345973,
    'chr8': 145138636,
    'chr9': 138394717,
    'chr10': 133797422,
    'chr11': 135086622,
    'chr12': 133275309,
    'chr13': 114364328,
    'chr14': 107043718,
    'chr15': 101991189,
    'chr16': 90338345,
    'chr17': 83257441,
    'chr18': 80373285,
    'chr19': 58617616,
    'chr20': 64444167,
    'chr21': 46709983,
    'chr22': 50818468,
}

def normalized_genomic_pos(row):
    chrom = row['chrom']
    chrom_length = chrom_lengths.get(chrom, 1)
    return row['genomic_pos'] / chrom_length

# Apply for transformations on all features
data['gc_content'] = data['sequence'].apply(gc_content)
data['at_content'] = data['sequence'].apply(at_content)
data['cpg_flag'] = data.apply(is_cpg_site, axis=1)
data['sequence_entropy'] = data['sequence'].apply(sequence_entropy)
data['is_transition'] = data.apply(lambda row: is_transition(row['ref'], row['alt']), axis=1)
data['genomic_pos_norm'] = data.apply(normalized_genomic_pos, axis=1)

# Left feature tensor
left_feature = data[['genomic_pos', 'gc_content', 'at_content', 'cpg_flag', 'sequence_entropy', 'is_transition', 'genomic_pos_norm']]

# Encoding all categorical columns
mutation_type_encoder = OneHotEncoder()
chromosome_encoder = OneHotEncoder()
ref_encoder = OneHotEncoder()
alt_encoder = OneHotEncoder()

# For mutation_type
mutation_type_encoder.fit(data[['mutation_type']])
mutation_type_data = pd.DataFrame(mutation_type_encoder.transform(data[['mutation_type']]).toarray())
mutation_type_data.columns = mutation_type_encoder.get_feature_names_out()

# For chrom
chromosome_encoder.fit(data[['chrom']])
chorm_data = pd.DataFrame(chromosome_encoder.transform(data[['chrom']]).toarray())
chorm_data.columns = chromosome_encoder.get_feature_names_out()

# For ref
ref_encoder.fit(data[['ref']])
ref_data = pd.DataFrame(ref_encoder.transform(data[['ref']]).toarray())
ref_data.columns = ref_encoder.get_feature_names_out()

# For alt
alt_encoder.fit(data[['alt']])
alt_data = pd.DataFrame(alt_encoder.transform(data[['alt']]).toarray())
alt_data.columns = alt_encoder.get_feature_names_out()

# Right features tensor
right_features = np.hstack((mutation_type_data, chorm_data, ref_data, alt_data))

# Build Tokenizer
def get_codon(seq, k=hyperparameters['k-mers']):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

vocab = {}

for seq in data['sequence']:
    for codons in get_codon(seq.lower()):
        if codons not in vocab:
            vocab[codons] = len(vocab)
        else:
            continue

def get_tensor(text):
    return [vocab[codons.lower()] for codons in get_codon(text)]

# Main features for training
x = data['sequence'].values
extra_features = np.hstack((left_feature, right_features))
y = data['label'].values

# Standarization for stable learning
feature_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(extra_features)

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, x, extra_features, y):
        self.x = x
        self.y = y
        self.extra_features = extra_features

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        seq_tensor = torch.tensor(get_tensor(self.x[index]), dtype=torch.long)
        features = torch.tensor(self.extra_features[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[index], dtype=torch.long)

        return seq_tensor, features, y_tensor
    
dataset = CustomDataset(x, scaled_features, y)

# Spliting data into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Dataloader for training and validation
train_loader = DataLoader(
    train_dataset,
    batch_size=hyperparameters['batch_size'],
    shuffle=True
)

test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'])

# Model Definition
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embed_dim, 2)) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class CNNTransformerHybrid(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_len, num_extra_features,
                 # Transformer specific params
                 num_heads=8, num_transformer_layers=6, ff_dim=2048,
                 # CNN specific params
                 cnn_out_channels=64,
                 # Common params
                 dropout=0.2):

        super(CNNTransformerHybrid, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.position_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_transformer_layers
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        cnn_output_len = max_len // 8
        flattened_cnn_size = cnn_out_channels * cnn_output_len

        combined_features_size = flattened_cnn_size + embed_dim + num_extra_features

        self.fc_layers = nn.Sequential(
            nn.Linear(combined_features_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_sequence, x_features):
        embeddings = self.embedding(x_sequence)

        transformer_input = self.position_encoding(embeddings)
        transformer_output = self.transformer_encoder(transformer_input)

        transformer_features = transformer_output.mean(dim=1)

        cnn_input = embeddings.permute(0, 2, 1)
        cnn_output = self.conv_layers(cnn_input)
        cnn_features = torch.flatten(cnn_output, 1)

        combined_features = torch.cat([transformer_features, cnn_features, x_features], dim=1)

        output = self.fc_layers(combined_features)
        return output
    
# Create object of model
model = CNNTransformerHybrid(
    vocab_size = len(vocab),
    embed_dim = hyperparameters['embed_dim'],
    num_classes = 2,
    max_len = hyperparameters['max_len'],
    dropout = hyperparameters['dropout'],
    num_heads = hyperparameters['num_heads'],
    num_transformer_layers = hyperparameters['num_transformer_layers'],
    ff_dim = hyperparameters['ff_dim'],
    cnn_out_channels = hyperparameters['cnn_out_channels'],
    num_extra_features = extra_features.shape[1],
)

if info['is_pre_training']:
    checkpoint = torch.load(f"/content/drive/MyDrive/{info['dir_name']}/model4thRun_epoch_60.pth")

    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {num_params}")

ce = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()

num_training_steps = len(train_loader) * hyperparameters['epochs']

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=hyperparameters['num_warmup_steps'],
    num_training_steps=num_training_steps
)

# Training Pipeline
def train(model, loader, ce, optimizer, scaler, scheduler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, features, y in loader:
        optimizer.zero_grad()
        x = x.to(device)
        features = features.to(device)
        y = y.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x, features)
            loss = ce(output, y)

        prediction = torch.argmax(output, dim=1)
        correct += (prediction == y).sum().item()
        total += len(x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        running_loss += loss.item() * len(x)

    accuracy = correct / total
    return (
        running_loss / len(loader.dataset),
        accuracy
    )

# Validation Pipeline
def validation(model, loader, ce):
    model.eval()

    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, features, y in loader:
            x = x.to(device)
            features = features.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(x, features)
                loss = ce(output, y)

            running_loss += loss.item() * len(x)

            prediction = torch.argmax(output, dim=1)
            correct += (prediction == y).sum().item()

            total += len(x)

    accuracy = correct / total

    return (
        running_loss / len(loader.dataset),
        accuracy
    )

# Whole Training Pipeline
patience = 10
best_val_loss = float('inf')
counter = 0
early_stop = False

train_loss_history = []
train_acc_history = []

val_loss_history = []
val_acc_history = []

save_dir = f"/content/drive/MyDrive/{info['dir_name']}"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, hyperparameters['epochs']+1):
    current_train_loss, current_train_acc = train(
          model,
          train_loader,
          ce,
          optimizer,
          scaler,
          scheduler
      )

    current_val_loss, current_val_acc = validation(
        model,
        test_loader,
        ce
    )

    train_loss_history.append(current_train_loss)
    train_acc_history.append(current_train_acc)

    val_loss_history.append(current_val_loss)
    val_acc_history.append(current_val_acc)

    print(f"Epoch ({epoch}/{hyperparameters['epochs']}): Train Loss = {current_train_loss:.4f}, Valitation Loss = {current_val_loss:.4f}, Train_acc = {current_train_acc:.4f}, Val_acc = {current_val_acc:.4f}")

    if epoch % 10 == 0:
        checkpoint_path = f"{save_dir}/model{info['run']}_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'encoders': {
                'mutation_type': mutation_type_encoder,
                'chromosome': chromosome_encoder,
                'ref': ref_encoder,
                'alt': alt_encoder
            },
            'feature_scaler': feature_scaler,
            'hyperparameters': hyperparameters,
            'vocab': vocab
        }, checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            counter = 0
            continue
        else:
            counter += 1
            print(f"No improvement in val loss Counter = {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered!")
                early_stop = True
                break

# Model Evaluation
def get_predictions_and_labels(model, loader):
    model.eval()
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for x, features, y in loader:
            x = x.to(device)
            features = features.to(device)
            y = y.to(device)

            yout = model(x, features)

            _, pred_mut = torch.max(yout, 1)

            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(pred_mut.cpu().numpy())

    return (
        (all_y_true, all_y_pred)
    )


(y_true, y_pred) = get_predictions_and_labels(model, test_loader)

print("\n" + "="*60)
print("Classification Report Summary")
print("="*60)

print("\n[1] Classification Report — Mutation Label")
print("-" * 60)
print(classification_report(y_true, y_pred))

print("="*60 + "\n")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45)


# Visualize Performance
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
