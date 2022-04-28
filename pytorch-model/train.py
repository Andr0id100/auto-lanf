import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from transformer import Transformer
from utils import create_mask
import time


en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

EN_FILE_PATH = "../europarl/truecased.en"
DE_FILE_PATH = "../europarl/truecased.de"


def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)


en_vocab = build_vocab_from_iterator(yield_tokens(open(
    EN_FILE_PATH), en_tokenizer), min_freq=100, specials=["<unk>", "<bos>", "<eos>", "<pad>"])
en_vocab.set_default_index(en_vocab["<unk>"])
de_vocab = build_vocab_from_iterator(yield_tokens(open(
    DE_FILE_PATH), de_tokenizer), min_freq=100, specials=["<unk>", "<bos>", "<eos>", "<pad>"])
de_vocab.set_default_index(de_vocab["<unk>"])


def data_process(en_path, de_path):
    raw_en_iter = iter(open(en_path, encoding="utf8"))
    raw_de_iter = iter(open(de_path, encoding="utf8"))

    data = []
    for (i, (raw_en, raw_de)) in enumerate(zip(raw_en_iter, raw_de_iter)):
        print(f"\r{i}", end="")

        en_tensor = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                                 dtype=torch.long)
        de_tensor = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                                 dtype=torch.long)
        data.append((en_tensor, de_tensor))
    return data


train_data = data_process(EN_FILE_PATH, DE_FILE_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (en_item, de_item) in data_batch:
        en_batch.append(
            torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch.append(
            torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    return en_batch, de_batch


BATCH_SIZE = 4
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0
    running_loss = 0

    for i, (src, tgt) in enumerate(iterator):
        if i % 10 == 9:
            print(f"{i}: {running_loss}")
            running_loss = 0
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, PAD_IDX)

        output = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask)

        optimizer.zero_grad()

        output = output.view(-1, output.shape[-1])
        tgt = tgt[1:].view(-1)

        loss = criterion(output, tgt)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        running_loss += loss.item()

    return epoch_loss / len(iterator)


transformer = Transformer(vocab_size=len(de_vocab)).to(device)
optimizer = optim.Adam(transformer.parameters())

N_EPOCHS = 5
CLIP = 1

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss = train(transformer, train_iter, optimizer, criterion, CLIP)

    print("Epoch:",         epoch)
    print("Train Loss:",    train_loss)
    print("Time Taken:",    time.time() - start_time)
    print()

torch.save(transformer, "saved_model.pth")
