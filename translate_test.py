import torch
from torch import nn
from torch.nn import Transformer
import jieba
import spacy
import math

# Load Spacy English model and define an English tokenizer
nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(text):
    """Tokenizes English text using the Spacy tokenizer."""
    return [token.text for token in nlp.tokenizer(text)]

# Jieba Chinese tokenizer
def jieba_tokenizer(text):
    """Tokenizes Chinese text using the Jieba tokenizer."""
    return list(jieba.cut(text))

# Mapping tokenizers to their respective languages
token_transform = {
    'zh': jieba_tokenizer,
    'en': spacy_tokenizer
}

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding for tokens
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        """Applies embedding and scales the result."""
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Positional encoding for sequence positions
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Adds positional encoding to the input embeddings."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer model for sequence-to-sequence translation
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        """Processes inputs through the encoder-decoder network."""
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outputs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                   src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outputs)

    def encode(self, src, src_mask):
        """Encodes the source sequence."""
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        """Decodes the target sequence."""
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

# Model and vocabulary loading paths
model_path = 'outputs/model.pth'
src_vocab_path = 'outputs/src_vocab.pth'
tgt_vocab_path = 'outputs/tgt_vocab.pth'

# Load the model and vocabularies
model = torch.load(model_path, map_location=device)
model.eval()
model = model.to(device)
src_vocab = torch.load(src_vocab_path)
tgt_vocab = torch.load(tgt_vocab_path)

# Vocabulary mapping
BOS_IDX = src_vocab['<bos>']
EOS_IDX = src_vocab['<eos>']
PAD_IDX = src_vocab['<pad>']
UNK_IDX = src_vocab['<unk>']

# Apply a sequence of transformations to the input text
def sequential_transforms(*transforms):
    """Combines multiple transforms into a single transformation."""
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func

# Convert tokens to tensor
def tensor_transform(token_ids):
    """Adds BOS and EOS tokens, then converts to tensor."""
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# Mapping transformations to their respective languages
text_transform = {
    'zh': sequential_transforms(jieba_tokenizer, src_vocab, tensor_transform),
    'en': sequential_transforms(spacy_tokenizer, tgt_vocab, tensor_transform)
}

# Generate a mask for the sequence
def generate_square_subsequent_mask(sz):
    """Generates a square mask for masking future tokens."""
    mask = torch.triu(torch.ones((sz, sz), device=device) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Create padding and source/target masks
def create_mask(src, tgt_input):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt_input.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.float)
    src_padding_mask = (src == PAD_IDX).type(torch.bool)
    tgt_padding_mask = (tgt_input == PAD_IDX).type(torch.bool)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Greedy decoding for translation
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(1)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys

# Translation function
def translate(model: torch.nn.Module, src_sentence: str):
    """Translates the input sentence from Chinese to English using the model."""
    model.eval()
    src = text_transform['zh'](src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(tgt_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace(" <eos>", "")

# Loop to get user input for translation
while True:
    src_sentence = input("请输入要翻译的句子（输入 'q' 退出）：")
    if src_sentence.lower() == 'q':
        break
    translated_sentence = translate(model, src_sentence)
    print("翻译结果:", translated_sentence)
