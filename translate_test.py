import torch
from torch import nn
from torch.nn import Transformer
import jieba
import spacy
import math

# 加载 Spacy 英语模型和定义英文分词器
nlp = spacy.load("en_core_web_sm")


def spacy_tokenizer(text):
    return [token.text for token in nlp.tokenizer(text)]


# jieba 中文分词器
def jieba_tokenizer(text):
    return list(jieba.cut(text))


# 分词器映射
token_transform = {
    'zh': jieba_tokenizer,
    'en': spacy_tokenizer
}

# 设定运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型定义
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outputs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                   src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outputs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


# 模型和词汇表的加载路径
model_path = 'outputs/model.pth'
src_vocab_path = 'outputs/src_vocab.pth'
tgt_vocab_path = 'outputs/tgt_vocab.pth'

# 加载模型和词汇表
model = torch.load(model_path, map_location=device)
model.eval()
model = model.to(device)
src_vocab = torch.load(src_vocab_path)
tgt_vocab = torch.load(tgt_vocab_path)

# 词汇映射
BOS_IDX = src_vocab['<bos>']
EOS_IDX = src_vocab['<eos>']
PAD_IDX = src_vocab['<pad>']
UNK_IDX = src_vocab['<unk>']


# 将 `text_transform` 从第二段代码复制过来
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# Tensor 转换
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


text_transform = {
    'zh': sequential_transforms(jieba_tokenizer, src_vocab, tensor_transform),
    'en': sequential_transforms(spacy_tokenizer, tgt_vocab, tensor_transform)
}


# 生成掩码
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), device=device) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt_input):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt_input.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.float)
    src_padding_mask = (src == PAD_IDX).type(torch.bool)
    tgt_padding_mask = (tgt_input == PAD_IDX).type(torch.bool)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# 贪婪解码函数
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


# 翻译函数
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform['zh'](src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(tgt_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace(" <eos>", "")


# 用户输入要翻译的句子
while True:
    src_sentence = input("请输入要翻译的句子（输入 'q' 退出）：")
    if src_sentence.lower() == 'q':
        break
    translated_sentence = translate(model, src_sentence)
    print("翻译结果:", translated_sentence)
