import argparse
import torch
import copy
import torch.nn as nn

from config import AMINO_ACIDS, device
import sys
import pdb
import numpy as np
from my_util import blosum, onehot, encode_amino, encode_sequence, seq2vec, vec2seq
import torch.optim as optim
from torch.optim import lr_scheduler

class VAE(nn.Module):
    def __init__(self, embed_size, hidden_size, latent_size, encode_method, beta=0.1, learned_size=20, min_len=8, max_len=15, bidirectional=True):
        super().__init__()

        self.min_len = min_len
        self.max_len = max_len

        self.encode_method = encode_method
                
        self.bidirectional = bidirectional
        
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = len(AMINO_ACIDS) + 1

        self.encoder_rnn = nn.LSTM(embed_size, 
                                   hidden_size, 
                                   bidirectional=self.bidirectional, 
                                   batch_first=True).to(device)
        
        # LSTM weights
        self.decoder_i = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Sigmoid() ).to(device)
        self.decoder_o = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Sigmoid() ).to(device)
        self.decoder_f = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Sigmoid() ).to(device)
        self.decoder_g = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Tanh() ).to(device)
        
        self.factor = 1 if not self.bidirectional else 2
        self.hidden_mean = nn.Linear(hidden_size * self.factor, latent_size).to(device)
        self.hidden_log = nn.Linear(hidden_size * self.factor, latent_size).to(device)
        self.latent2hidden = nn.Linear(latent_size, hidden_size).to(device)
        
        #self.embed2hidden = nn.Linear(embed_size, hidden_size)
        
        self.hidden2out = nn.Sequential(
                               nn.Linear(embed_size + hidden_size + latent_size, hidden_size),
                               nn.ReLU(),
                               nn.Linear(hidden_size, self.output_size)).to(device)

        self.beta = beta
        self.loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, sequences):
        hidden = self.encode(sequences)

        z, kl_loss = self.sample(hidden)

        amino_loss, acc = self.decode(sequences, z)

        loss = amino_loss + self.beta * kl_loss
        
        return loss, amino_loss, kl_loss, acc, z

    
    def encode(self, input_sequence):
        batch_size = len(input_sequence)
        
        length = torch.LongTensor([len(seq) for seq in input_sequence])
        
        h0 = torch.zeros(self.factor, len(input_sequence), self.hidden_size).float().to(device)
        c0 = torch.zeros(self.factor, len(input_sequence), self.hidden_size).float().to(device)
        
        peptide_encodings = torch.tensor(encode_sequence(input_sequence, self.encode_method)).float().to(device)
        
        packed_encodings = torch.nn.utils.rnn.pack_padded_sequence(peptide_encodings, batch_first=True, lengths=length, enforce_sorted=False).to(device)
        
        _, (hidden, _) = self.encoder_rnn(packed_encodings, (h0, c0))
        
        if self.bidirectional:
            hidden = hidden.transpose(0, 1).flatten(start_dim=-2)
        else:
            hidden = hidden.squeeze()

        return hidden

    def sample(self, hidden):
        # REPARAMETERIZATION
        mean = self.hidden_mean(hidden)
        log = self.hidden_log(hidden)
        std = torch.exp(0.5 * log)

        z = torch.randn([hidden.shape[0], self.latent_size]).to(device)
        z = z * std + mean

        kl_loss = -0.5 * torch.sum(1.0 + log - mean * mean - torch.exp(log)) / hidden.shape[0]
        return z, kl_loss

    def _decoder_lstm(self, x, h, c):
        i = self.decoder_i( torch.cat([x, h], dim=-1) )
        o = self.decoder_o( torch.cat([x, h], dim=-1) )
        f = self.decoder_f( torch.cat([x, h], dim=-1) )
        g = self.decoder_g( torch.cat([x, h], dim=-1) )
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

    def decode(self, input_sequences, z):
        hidden = self.latent2hidden(z)
        cell = torch.zeros_like(hidden).to(device)

        init_x_vecs = torch.zeros((z.shape[0], self.embedding_size)).to(device)
        
        amino_hiddens, targets = [], []
        
        for i, seq in enumerate(input_sequences):
            input_sequences[i] = seq[::-1]
        
        seq_arrays = seq2vec(input_sequences)
        
        for t in range(self.max_len):
            
            tokens, nexts, nonstop_idxs = [], [], []
            for i, seq in enumerate(input_sequences):
                if t <= len(seq):
                    if t > 0: 
                        tokens.append(seq[t-1])
                    
                    if t < len(seq):
                        if t != 0: nonstop_idxs.append(len(tokens) - 1)
                        nexts.append(seq_arrays[i][t])
                    else:
                        nexts.append(0)
            
            if t == 0: nonstop_idxs = [i for i in range(len(input_sequences))]
            targets.extend(nexts)
            
            if t == 0:
                x_vecs = init_x_vecs
            else:
                x_vecs = torch.tensor(encode_amino(tokens, self.encode_method)).float().to(device)
                  
            amino_hiddens.append( torch.cat( (x_vecs, hidden.clone(), z), dim=-1))
            
            new_h, new_c = self._decoder_lstm(x_vecs, hidden, cell)
            
            if len(nonstop_idxs) == 0: break
            
            nonstop_idxs = torch.LongTensor(nonstop_idxs).to(device)
            z = torch.index_select(z, 0, nonstop_idxs)
            hidden = torch.index_select(new_h, 0, nonstop_idxs)
            cell   = torch.index_select(new_c, 0, nonstop_idxs)
        
        amino_hiddens = torch.cat(amino_hiddens, dim=0)
        amino_scores = self.hidden2out(amino_hiddens).squeeze(dim=1)
        
        targets = torch.LongTensor(targets).to(device)
        loss = self.loss(amino_scores, targets) / len(input_sequences)
        _, amino = torch.max(amino_scores, dim=1)
        
        amino_acc = torch.eq(amino, targets).float()
        amino_acc = torch.sum(amino_acc) / targets.shape[0]
        
        return loss, amino_acc
        
    def generate(self, z):
        hidden = self.latent2hidden(z)
        cell   = torch.zeros_like(hidden).to(device)
        
        x_vecs = torch.zeros((z.shape[0], self.embedding_size)).to(device)
        padded_sequence = torch.zeros(z.shape[0], self.max_len, dtype=torch.long).to(device)
        
        idxs = torch.arange(z.shape[0]).to(device)
        
        for i in range(self.max_len):
            amino_hidden = torch.cat((x_vecs, hidden, z), dim=-1)
            
            amino_scores = self.hidden2out(amino_hidden)

            if i <= self.min_len:
                amino_scores[:, 0] = torch.ones(z.shape[0]) * -1000
            
            _, out = torch.max(amino_scores, dim=1)
            
            nonstop_idxs = torch.arange(hidden.shape[0]).to(device)
            if i > self.min_len:
                nonstop_idxs = torch.nonzero(out).squeeze(dim=1).to(device)
                
                out = torch.index_select(out, 0, nonstop_idxs)
                idxs = torch.index_select(idxs, 0, nonstop_idxs)
            
            if nonstop_idxs.shape[0] == 0: break
            padded_sequence[idxs, i] = out
            
            if self.encode_method == "blosum":
                out_vec = blosum(out.cpu().detach().numpy())
            else:
                out_vec = onehot(out.cpu().detach().numpy())
            
            out_vec = torch.tensor(out_vec).float().to(device)
            
            new_hidden = torch.index_select(hidden, 0, nonstop_idxs)
            z          = torch.index_select(z, 0, nonstop_idxs)
            new_cell   = torch.index_select(cell, 0, nonstop_idxs)
            x_vecs     = torch.index_select(x_vecs, 0, nonstop_idxs)
            hidden, cell = self._decoder_lstm(x_vecs, new_hidden, new_cell)
            x_vecs = out_vec
            
            
        seqs = vec2seq(padded_sequence)
        seqs = [seq[::-1] for seq in seqs]
        return seqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_method", type=str, default="blosum")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--stop_criteria", type=int, default=20)
    parser.add_argument("--max_step", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--print_iter", type=int, default=20)
    parser.add_argument("--save_iter", type=int, default=50000)
    
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--anneal_iter", type=int, default=5000)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=0.0005)

    parser.add_argument("--beta_anneal_iter", type=int, default=10000)
    parser.add_argument("--max_beta", type=float, default=0.5)
    parser.add_argument("--step_beta", type=float, default=0.1)

    parser.add_argument("--clip_norm", type=float, default=50.0)
    args = parser.parse_args()

    embed_methods = ['blosum', 'onehot', 'deep']

    embedding_size = 0
    for method in embed_methods:
        if method in args.embed_method: embedding_size += 20

    model = VAE(embedding_size, args.hidden_size, args.latent_size, \
                args.embed_method, beta=args.beta)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate) 
    
    num, iter_num, accs, losses, beta_losses, amino_losses = 0, 0, 0, 0, 0, 0

    train_peptide_len = np.random.choice(np.arange(8, 15), 2000000)
    train_peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in train_peptide_len]

    print("with %s device" % (device))
    print("with update rate: %.1f" % (args.rate))
    beta = args.beta
    while num < args.stop_criteria and iter_num < args.max_step:
        iter_num += 1
        
        #update_idxs = np.random.choice(args.batch_size, int(args.rate * args.batch_size))
        
        #update_train_peptide_len = np.random.choice(np.arange(8, 15), len(update_idxs))
        #update_train_peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in update_train_peptide_len]

        batch_peptides_idx = np.random.choice(len(train_peptides), args.batch_size)
        batch_peptides = [train_peptides[i] for i in batch_peptides_idx]
        
        #for i, idx in enumerate(update_idxs): train_peptides[idx] = update_train_peptides[i]
        
        with torch.autograd.set_detect_anomaly(True):
            model.zero_grad()
            
            loss, amino_loss, beta_loss, acc, _ = model(batch_peptides)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
       
        accs = accs + float(acc)
        losses = losses + float(loss)
        beta_losses = beta_losses + float(beta_loss)
        amino_losses = amino_losses + float(amino_loss)
        
        if iter_num % args.print_iter == 0:
            test_peptide_len = np.random.choice(np.arange(8, 15), args.batch_size)
            test_peptides = ["".join(list(np.random.choice(AMINO_ACIDS, plen))) for plen in test_peptide_len]
            
            _, _, _, new_acc, _ = model(test_peptides)
            
            accs /= args.print_iter
            losses /= args.print_iter
            beta_losses /= args.print_iter
            amino_losses /= args.print_iter
            print("step: %d; accuracy: %.4f; test_accuracy: %.4f, loss: %.4f; beta: %.4f, amino: %.4f" % (iter_num, accs, new_acc, losses, beta_losses, amino_losses))
            sys.stdout.flush()
            accs = 0
            losses= 0
            beta_losses = 0
            amino_losses = 0
        
            #if new_acc > 0.98:
            #    num += 1
            #else:
            #    num = 0

        if iter_num % args.save_iter == 0:
            torch.save(model.state_dict(), "%s_step%d" % (args.model_path, iter_num))

        if iter_num % args.beta_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)
            model.beta = beta
            print("beta value: %.2f" % (beta))
            
        if iter_num % args.anneal_iter == 0 and scheduler.get_lr()[0] > args.min_lr:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])


    torch.save(model.state_dict(), "%s.pt" % (args.model_path))

