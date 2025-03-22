import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models import GFK
from utils import load_dataset, edgeindex_construct

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, adj_t)


class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.dev}' if torch.cuda.is_available() else 'cpu')
        
        data = np.load(f'data/{args.dataset}/{args.dataset}.npz')
        self.edge_index = data['edge_index']
        self.features = torch.FloatTensor(data['feats']).to(self.device)
        self.labels = torch.LongTensor(data['labels']).squeeze().to(self.device)
        
        self.adj = self.build_adjacency()
        self.split_idx = self.load_splits()
        
        self.model_config = {
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.5,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'K': 5 
        }

    def build_adjacency(self):
        edge_tensor = torch.LongTensor(self.edge_index).to(self.device)
        adj = SparseTensor(row=edge_tensor[0], col=edge_tensor[1],
                          sparse_sizes=(self.labels.size(0), self.labels.size(0)))
        adj = adj.to_symmetric()
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.view(-1,1) * adj * deg_inv_sqrt.view(1,-1)

    def load_splits(self):
        split_idx = np.load(f'data/splits/{self.args.dataset}-splits.npy', allow_pickle=True)[0]
        return {k: torch.LongTensor(v).to(self.device) for k,v in split_idx.items()}

    def run_gcn(self):
        model = GCN(
            in_channels=self.features.size(1),
            hidden_channels=self.model_config['hidden_dim'],
            out_channels=int(self.labels.max())+1,
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.model_config['lr'],
                                   weight_decay=self.model_config['weight_decay'])
        
        return self._train_model(model, optimizer, is_gcn=True)

    def run_gfk(self):
        LP = edgeindex_construct(self.edge_index, self.labels.size(0))[0]
        propagated_features, dim = load_dataset(LP, self.features, 
                                            self.model_config['K'], tau=0.5)
        num_nodes = self.labels.size(0)
        propagated_features = propagated_features.view(num_nodes, -1, dim)
        model = GFK(
            level=self.model_config['K'],
            nfeat=dim,
            nlayers=self.model_config['num_layers'],
            nhidden=self.model_config['hidden_dim'],
            nclass=int(self.labels.max())+1,
            dropoutC=self.model_config['dropout'],
            dropoutM=self.model_config['dropout'],
            bias=self.args.bias
        ).to(self.device)
        
        optimizer = torch.optim.AdamW([
            {'params': model.mlp.parameters(), 'weight_decay': self.model_config['weight_decay']},
            {'params': model.comb.parameters(), 'weight_decay': self.model_config['weight_decay']}
        ], lr=self.model_config['lr'])
        
        return self._train_model(model, optimizer, features=propagated_features)

    def _train_model(self, model, optimizer, features=None, is_gcn=False):
        history = {'loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
        best_acc = 0
        model.train()
        
        train_idx = self.split_idx['train']
        
        for epoch in range(1, self.args.epochs+1):
            optimizer.zero_grad()
            
            if is_gcn:
                out = model(self.features, self.adj)
            else:
                out = model(features)
            
            loss = F.cross_entropy(out[train_idx], self.labels[train_idx])
            
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0 or epoch == self.args.epochs:
                model.eval()
                with torch.no_grad():
                    if is_gcn:
                        full_out = model(self.features, self.adj)
                    else:
                        full_out = model(features)
                    
                    train_acc = self._accuracy(full_out, 'train')
                    val_acc = self._accuracy(full_out, 'valid')
                    test_acc = self._accuracy(full_out, 'test')
                    
                    history['loss'].append(loss.item())
                    history['train_acc'].append(train_acc)
                    history['val_acc'].append(val_acc)
                    history['test_acc'].append(test_acc)
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), 'best_model.pt')
                        
                model.train()
                print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} '
                      f'| Train: {train_acc:.2f}% | Val: {val_acc:.2f}% '
                      f'| Test: {test_acc:.2f}%')
        
        model.load_state_dict(torch.load('best_model.pt'))
        model.eval()
        with torch.no_grad():
            final_test = self._accuracy(model(self.features, self.adj) if is_gcn 
                                      else model(features), 'test')
        
        return history, final_test

    def _accuracy(self, outputs, split):
        preds = outputs.argmax(dim=-1)
        return (preds[self.split_idx[split]] == self.labels[self.split_idx[split]]).float().mean().item() * 100

    def visualize(self, gcn_history, gfk_history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.plot(gcn_history['loss'], label='GCN')
        plt.plot(gfk_history['loss'], label='GFK')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(132)
        plt.plot(gcn_history['val_acc'], label='GCN')
        plt.plot(gfk_history['val_acc'], label='GFK') 
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        
        plt.subplot(133)
        plt.bar(['GCN', 'GFK'], 
                [gcn_history['test_acc'][-1], gfk_history['test_acc'][-1]])
        plt.ylabel('Final Test Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig('comparison.png')
        plt.show()

if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.dataset = 'ogbn-arxiv'
            self.epochs = 100
            self.dev = 0
            self.runs = 3
            self.bias = True
    
    experiment = ExperimentRunner(Args())
    
    print("Running GCN Experiment...")
    gcn_results, gcn_final = experiment.run_gcn()
    
    print("\nRunning GFK Experiment...")
    gfk_results, gfk_final = experiment.run_gfk()
    
    experiment.visualize(gcn_results, gfk_results)
    
    print(f"\nGCN Final Test Accuracy: {gcn_final:.2f}%")
    print(f"GFK Final Test Accuracy: {gfk_final:.2f}%")
    print(f"Accuracy Improvement: {(gfk_final - gcn_final):.2f}%")
