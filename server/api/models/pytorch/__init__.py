import torch
import spacy
from torch import nn
from torch.nn import functional as F
from models import PYTORCH_AHD_MODEL_PATH, PYTORCH_AHD_VOCAB, device, CLASSES, PredictionType

# Tokens 
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

print(PYTORCH_AHD_VOCAB[PAD_TOKEN], PYTORCH_AHD_VOCAB[UNK_TOKEN], PYTORCH_AHD_VOCAB['the'])

# Tokenizer
print(" ✅ LOADING TOKENIZER FROM SPACY(en_core_web_sm)!\n")
spacy_en = spacy.load('en_core_web_sm')
print(" ✅ LOADING TOKENIZERS DONE!\n")

def tokenize_en(sent: str) -> list:
    return [tok.text for tok in spacy_en.tokenizer(sent)]

def text_pipeline(x:str)->list:
    values = list()
    tokens = tokenize_en(x.lower())
    for token in tokens:
        try:
            v = PYTORCH_AHD_VOCAB[token]
        except:
            v = PYTORCH_AHD_VOCAB[UNK_TOKEN]
            values.append(v)
    return values

class AHDCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_filters, filter_sizes, output_size, 
            dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                            out_channels = n_filters, 
                                            kernel_size = (fs, embedding_size)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):  
        embedded = self.embedding(text)    
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1)) 
        return self.fc(cat)
   
print(" ✅ LOADING PYTORCH AHD MODEL!\n") 
# Hyper params
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
INPUT_DIM = len(PYTORCH_AHD_VOCAB) 
PAD_IDX = PYTORCH_AHD_VOCAB[PAD_TOKEN]

# Model instance
ahd_model = AHDCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, 
                FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX).to(device)

ahd_model.load_state_dict(torch.load(PYTORCH_AHD_MODEL_PATH, map_location=device))
print(" ✅ LOADING PYTORCH AHD MODEL DONE!\n")

def preprocess_text(text, max_len=50, padding="pre"):
    assert padding=="pre" or padding=="post", "the padding can be either pre or post"
    text_holder = torch.zeros(max_len, dtype=torch.int32) # fixed size tensor of max_len with <pad> = 0
    processed_text = torch.tensor(text_pipeline(text), dtype=torch.int32)
    
    pos = min(max_len, len(processed_text))
    if padding == "pre":
        text_holder[:pos] = processed_text[:pos]
    else:
        text_holder[-pos:] = processed_text[-pos:]
    text_list= text_holder.unsqueeze(dim=0)
    return text_list

def predict_homour(sent: str, model):
    model.eval()
    tensor = preprocess_text(sent)
    pred = torch.sigmoid(model(tensor.to(device))).item()
    
    label = 1 if pred >=0.5 else 0
    probability = float(round(pred, 3)) if pred >= 0.5 else float(round(1 - pred, 3))
    return PredictionType(label=label, 
                          probability=probability, 
                          class_= CLASSES[label]
                          )