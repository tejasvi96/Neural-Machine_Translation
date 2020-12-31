# Text only Translation Model
# With Attention Model
import torch
from tqdm import tqdm
import torch.nn as nn
import os,torch

import numpy as np
from loguru import logger

import unicodedata,re
from torch.utils.data import TensorDataset, DataLoader
logger.add("logfile.txt")
dir_path="/storage"

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

train_fname="/hindi-visual-genome-train.txt"
val_fname="/hindi-visual-genome-dev.txt"
test_fname="/hindi-visual-genome-test.txt"
hidden_size=1280
embedding_size=300
lr=0.01
num_epochs=80
MAX_LENGTH=20
batch_size=32

# To store the output sentences produced
output_file='outputs.txt'

logger.info(train_fname)
logger.info(val_fname)
logger.info(test_fname)
logger.info(hidden_size)
logger.info(embedding_size)
logger.info(lr)
logger.info(num_epochs)
logger.info(MAX_LENGTH)


def get_sents(fname):
    cwd=dir_path
    filename=cwd+fname
    with open(filename,'r',encoding='utf-8') as fp:
        data=fp.readlines()
    hindi_sents=[]
    eng_sents=[]
    for i in data:
        temp=i.split("\t")[6]
        hindi_sents.append(temp)
        temp=i.split("\t")[5]
        eng_sents.append(temp)
    
    hindi_sents=[i.replace("\n","") for i in hindi_sents ]
    eng_sents=[i.replace("\n","") for i in eng_sents ]
    return hindi_sents,eng_sents



train_hindi_sents,train_eng_sents=get_sents(train_fname)
val_hindi_sents,val_eng_sents=get_sents(val_fname)
test_hindi_sents,test_eng_sents=get_sents(test_fname)


pad_token = 0
SOS_token = 1
EOS_token = 2

unk_token=3
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "SOS", 2: "EOS",0:"<pad>",3:"<unk>"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

Eng=Lang("English")
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

for sent in train_eng_sents:
    sent=normalizeString(sent)
    Eng.addSentence(sent)
for sent in val_eng_sents:
    sent=normalizeString(sent)
    Eng.addSentence(sent)

Hin=Lang("Hindi")
for sent in train_hindi_sents:
    Hin.addSentence(sent)
for sent in val_hindi_sents:
    Hin.addSentence(sent)

hindi_vectors=torch.zeros((Hin.n_words,embedding_size))
eng_vectors=torch.zeros((Eng.n_words,embedding_size))
import fasttext
import fasttext.util
fasttext.util.download_model('en') 
fasttext.util.download_model('hi') 
ft_hi = fasttext.load_model('cc.hi.300.bin')
ft_en= fasttext.load_model('cc.en.300.bin')

for word in Hin.word2index.keys():
    hindi_vectors[Hin.word2index[word]]=torch.tensor( ft_hi.get_word_vector(word) )
for word in Eng.word2index.keys():
    eng_vectors[Eng.word2index[word]]=torch.tensor( ft_en.get_word_vector(word) )


start_vec=np.random.rand((embedding_size))
eos_vec=np.random.rand((embedding_size))
unk_vec=np.random.rand((embedding_size))
start_vec=torch.tensor(start_vec.reshape(-1,embedding_size))
eos_vec=torch.tensor(eos_vec.reshape(-1,embedding_size))
unk_vec=torch.tensor(unk_vec.reshape(-1,embedding_size))
hindi_vectors[SOS_token]=start_vec
hindi_vectors[EOS_token]=eos_vec
hindi_vectors[unk_token]=unk_vec
eng_vectors[SOS_token]=start_vec
eng_vectors[EOS_token]=eos_vec
eng_vectors[unk_token]=unk_vec

def word_tokenize(lang,sent):
    if(lang.name=="English"):
        sent=normalizeString(sent)
    tokenized_sent=[]
    tokenized_sent.append(SOS_token)
    for word in sent.split():
        if word in lang.word2index.keys():
            tokenized_sent.append(lang.word2index[word])
        else:
            tokenized_sent.append(3)
    
    tokenized_sent.append(EOS_token)
    l=len(tokenized_sent)
    if(l!=MAX_LENGTH):
        if(l<MAX_LENGTH):
            tokenized_sent=tokenized_sent+([pad_token]*(MAX_LENGTH-l))
        else:
            tokenized_sent=tokenized_sent[:MAX_LENGTH-1]+[EOS_token]
#     tokenized_sent.append(1)
    
    return tokenized_sent


def get_lengths(sent):
    return (MAX_LENGTH)
#     return min(MAX_LENGTH,len(sent)+2)

train_eng_sents_lengths=[get_lengths(sent) for sent in train_eng_sents]
val_eng_sents_lengths=[get_lengths(sent) for sent in val_eng_sents]
train_eng_sents_tokenized=[word_tokenize(Eng,sent) for sent in train_eng_sents]
val_eng_sents_tokenized=[word_tokenize(Eng,sent) for sent in val_eng_sents]
train_hin_sents_tokenized=[word_tokenize(Hin,sent) for sent in train_hindi_sents]
val_hin_sents_tokenized=[word_tokenize(Hin,sent) for sent in val_hindi_sents]

train_hin_sents_lengths=[get_lengths(sent) for sent in train_hindi_sents]
val_hin_sents_lengths=[get_lengths(sent) for sent in val_hindi_sents]

test_eng_sents_tokenized=[word_tokenize(Eng,sent) for sent in test_eng_sents]
test_hin_sents_tokenized=[word_tokenize(Hin,sent) for sent in test_hindi_sents]
test_eng_sents_lengths=[get_lengths(sent) for sent in test_eng_sents]
test_hin_sents_lengths=[get_lengths(sent) for sent in test_hindi_sents]

from torch.utils.data import TensorDataset, DataLoader
import numpy as np
train_data=TensorDataset(torch.from_numpy(np.array(train_eng_sents_tokenized,dtype=np.long)),torch.from_numpy(np.array(train_eng_sents_lengths)),torch.from_numpy(np.array(train_hin_sents_tokenized,dtype=np.long)),torch.from_numpy(np.array(train_hin_sents_lengths)))
val_data=TensorDataset(torch.from_numpy(np.array(val_eng_sents_tokenized,dtype=np.long)),torch.from_numpy(np.array(val_eng_sents_lengths)),torch.from_numpy(np.array(val_hin_sents_tokenized,dtype=np.long)),torch.from_numpy(np.array(val_hin_sents_lengths)))
test_data=TensorDataset(torch.from_numpy(np.array(test_eng_sents_tokenized,dtype=np.long)),torch.from_numpy(np.array(test_eng_sents_lengths)),torch.from_numpy(np.array(test_hin_sents_tokenized,dtype=np.long)),torch.from_numpy(np.array(test_hin_sents_lengths)))

trainloader=DataLoader(train_data,batch_size=batch_size)
valloader=DataLoader(val_data,batch_size=batch_size)
testloader=DataLoader(test_data,batch_size=1)

logger.info("Data preprocessing done")

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_sent_len, embedding_dim):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_sent_len = max_sent_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)

    def forward(self, source):
        source = self.embedding(source)
        enc_outputs = torch.zeros(self.max_sent_len, source.size(0), self.hidden_size).to(device)
        h0 = torch.zeros(1, source.size(0), self.hidden_size).to(device)  # encoder hidden state = (1, BATCH_SIZE, HIDDEN_SIZE)
        for k in range(source.size(1)):  
            _, h0 = self.gru(source[:, k].unsqueeze(0), h0)
            enc_outputs[k, :] = h0.squeeze()
        return enc_outputs

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.Linear(hidden_size + hidden_size, 1)
        self.gru = nn.GRU(hidden_size + embedding_dim, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim = 1)
        self.relu = nn.ReLU()

    def forward(self, decoder_input, current_hidden_state, encoder_outputs):

        decoder_input = self.embedding(decoder_input)    # (BATCH_SIZE, EMBEDDING_DIM)
        aligned_weights = torch.randn(encoder_outputs.size(0), encoder_outputs.size(1)).to(self.device)

        for i in range(encoder_outputs.size(0)):
            aligned_weights[i] = self.attention(torch.cat((current_hidden_state, encoder_outputs[i].unsqueeze(0)), dim = -1)).squeeze()

        aligned_weights = self.softmax(aligned_weights)   # (BATCH_SIZE, HIDDEN_STATE * 2)
        aligned_weights = aligned_weights.view(aligned_weights.size(1), aligned_weights.size(0))

        context_vector = torch.bmm(aligned_weights.unsqueeze(1), encoder_outputs.view(encoder_outputs.size(1), encoder_outputs.size(0), encoder_outputs.size(2)))

        x = torch.cat((context_vector.squeeze(1), decoder_input), dim = 1).unsqueeze(0)
        x = self.relu(x)
        x, current_hidden_state = self.gru(x, current_hidden_state)
        x = self.log_softmax(self.dense(x.squeeze(0)))
        return x, current_hidden_state, aligned_weights

class AttenS2S(nn.Module):
    def __init__(self, encoder, decoder, max_sent_len, device):
        super(AttenS2S, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_sent_len = max_sent_len

    def forward(self, source, target, tf_ratio = .5):
        enc_outputs = self.encoder(source)
        dec_outputs = torch.zeros(target.size(0), target.size(1), self.decoder.vocab_size).to(self.device)
        dec_input = target[:, 0]
        dec_h0 = torch.zeros(1, dec_input.size(0), self.encoder.hidden_size).to(device)
        weights = torch.zeros(target.size(1), target.size(0), target.size(1))   # (TARGET_LEN, BATCH_SIZE, SOURCE_LEN)
        for k in range(target.size(1)):
            out, dec_h0, w = self.decoder(dec_input, dec_h0, enc_outputs)
            weights[k, :, :] = w
            dec_outputs[:, k] = out
            if np.random.choice([True, False], p = [tf_ratio, 1-tf_ratio]):
                dec_input = target[:, k]
            else:
                dec_input = out.argmax(1).detach()

        return dec_outputs, weights


encoder = Encoder(Eng.n_words, hidden_size,MAX_LENGTH, embedding_size).to(device)
decoder = Decoder(Hin.n_words, hidden_size, embedding_size,device).to(device)
seq2seq = AttenS2S(encoder, decoder, MAX_LENGTH, device).to(device)
criterion = nn.NLLLoss(reduction='sum',ignore_index=pad_token)
optimizer = torch.optim.Adam(seq2seq.parameters(), lr = lr)

logger.info("Model defintion done")
# num_epochs=10
loss_value = []
for epoch in tqdm(range(num_epochs)):
    current_loss = 0
    items=0
    for i, (data) in tqdm(enumerate(trainloader),total=int(len(trainloader)/trainloader.batch_size)):
        x, y  = data[0].to(device), data[2].to(device)
        items+=data[0].shape[0]
        outputs,_ = seq2seq(x, y)
        loss = criterion(outputs.resize(outputs.size(0) * outputs.size(1), outputs.size(-1)), y.resize(y.size(0) * y.size(1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
    loss_value.append(current_loss)
    print(current_loss/items)


import matplotlib.pyplot as plt
plt.plot(range(1, num_epochs+1), loss_value, 'r-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

logger.info(loss_value)

logger.info("Making the predictions on the testdata")

def sentence_decode(inp_tensor,Lang):
    inp_tensor=inp_tensor.T
    words=[]
    for i in inp_tensor:
        words.append(Lang.index2word[i.item()])
        if i.item()==0:
            break
    if "SOS" in words:
        words.remove("SOS")
    if "<pad>" in words:
        words.remove("<pad>")    
    return words

predictions = []
for i, (data) in enumerate(testloader):
    with torch.no_grad():
        x, y  = data[0].to(device), data[2].to(device)
        outputs,_ = seq2seq(x, y)
        for output in outputs:
            _, indices = output.max(-1)
            predictions.append(indices.detach().cpu().numpy())


predicted_sents=[sentence_decode(predictions[i],Hin) for i in range(len(predictions))]
actual_sents=[sentence_decode(test_data[i][2],Hin) for i in range(len(test_data))]

with open(output_file,'w',encoding='utf-8') as fp:
    for i in predicted_sents:
        fp.write("".join(predicted_sents[i])+"\n")


from nltk.translate import bleu_score
score=[]
for i in range(len(predicted_sents)):
    r=predicted_sents[i]
    s=actual_sents[i]
    try:
        score.append(bleu_score.corpus_bleu(r,s))
    except:
        pass
     
print(sum(score)/len(score))
logger.info(sum(score)/len(score))