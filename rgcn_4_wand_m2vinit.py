import copy
import argparse
import builtins
import torch
import wandb
print(torch.cuda.is_available())
print("look just above")
import torch.nn.functional as F
from torch.nn import Parameter, ModuleDict, ModuleList, Linear, ParameterDict
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
import os 
print("everything is loaded")
os.system("nvidia-smi")

# load the embedding: 

def load_pretrained_embeddings(node_types, embedding_dir, embedding_dim):
    pre_trained_embs = {}
    for node_type in node_types:
        embedding_path = os.path.join(embedding_dir, f"{node_type}.pt")
        print(embedding_path)
        if os.path.exists(embedding_path):
            print(f"Loading embeddings for {node_type} from {embedding_path}")
            emb = torch.load(embedding_path)
            if emb.size(1) == embedding_dim:
                pre_trained_embs[node_type] = emb
            else:
                raise ValueError(f"Embedding dimension mismatch for {node_type}: "
                                 f"Expected {embedding_dim}, got {emb.size(1)}")
        else:
            print(f"No pretrained embeddings found for {node_type}. Initializing randomly.")
    return pre_trained_embs




class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels # length of embedding vector ( defined )data.x_dict['paper'].size(-1) ( 128 for word2vec)
        self.out_channels = out_channels # dataset.num_classes

        # `ModuleDict` does not allow tuples :(
        self.rel_lins = ModuleDict({
            f'{key[0]}_{key[1]}_{key[2]}': Linear(in_channels, out_channels,bias=False)
            for key in edge_types
        })

        self.root_lins = ModuleDict({
            key: Linear(in_channels, out_channels, bias=True)
            for key in node_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x = x_dict[key[0]]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean')) # normalization constant.
            out_dict[key[2]].add_(out)

        return out_dict


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, edge_types, pre_trained_embs = None):
        super(RGCN, self).__init__()

        node_types = list(num_nodes_dict.keys())

        self.embs = ParameterDict({
            key: Parameter(pre_trained_embs[key].clone()) if pre_trained_embs and key in pre_trained_embs # Take the metapath2vec embedding, it not present, take word to vec + random embedding
            else Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        self.convs = ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, node_types, edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, node_types,
                         edge_types))
        self.convs.append(
            RGCNConv(hidden_channels, out_channels, node_types, edge_types))

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            if emb.requires_grad:
                torch.nn.init.xavier_uniform_(emb) # perhaps here you initialize embedding of nodes whose embedding is  not predefined uniform 
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        x_dict = copy.copy(x_dict)
        for key, emb in self.embs.items():
            x_dict[key] = emb

        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, adj_t_dict)
            for key, x in x_dict.items():
                x_dict[key] = F.relu(x)
                x_dict[key] = F.dropout(x, p=self.dropout,
                                        training=self.training)
        return self.convs[-1](x_dict, adj_t_dict)


def train(model, x_dict, adj_t_dict, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x_dict, adj_t_dict)['paper'].log_softmax(dim=-1)
    loss = F.nll_loss(out[train_idx], y_true[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, x_dict, adj_t_dict, y_true, split_idx, evaluator):
    model.eval()

    out = model(x_dict, adj_t_dict)['paper']
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-MAG (Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64) # set dat boi up to 349 
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50) # was 50
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--save_model', action='store_true', help="Save model to file") # save the model
    parser.add_argument('--embedding_dir', type=str, required=True,help="Path to the directory containing pre-trained embeddings") # the path to embedding dir:
    parser.add_argument('--embedding_dim', type=int, default=128, help="Dimension of the embeddings") # the dimension of the 
    args = parser.parse_args()
    print(args)
    wandb.init(project="ogbn-mag", config=vars(args))

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # returnerer N ( no), for hver terminal spørgsmål så vi undgår EOFError
    builtins.input = lambda _: 'N'
    print("builtins does not fuks up")

    dataset = PygNodePropPredDataset(name='ogbn-mag') # added preprocessed=True, and deleted again
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    
    
    wandb.config.update({"dataset": "ogbn-mag"})

    # We do not consider those attributes for now.
    data.node_year_dict = None
    data.edge_reltype_dict = None

    print(data)

    # Convert to new transposed `SparseTensor` format and add reverse edges.
    data.adj_t_dict = {}
    for keys, (row, col) in data.edge_index_dict.items():
        sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
        adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
        # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
        if keys[0] != keys[2]:
            data.adj_t_dict[keys] = adj.t()
            data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
        else:
            data.adj_t_dict[keys] = adj.to_symmetric()
    data.edge_index_dict = None

    x_types = list(data.x_dict.keys())
    edge_types = list(data.adj_t_dict.keys())
    pre_trained_embs = load_pretrained_embeddings(data.num_nodes_dict.keys(), args.embedding_dir, args.embedding_dim)

    model = RGCN(data.x_dict['paper'].size(-1), args.hidden_channels,
                 dataset.num_classes, args.num_layers, args.dropout,
                 data.num_nodes_dict, x_types, edge_types, pre_trained_embs)

    data = data.to(device)
    model = model.to(device)
    train_idx = split_idx['train']['paper'].to(device)

    evaluator = Evaluator(name='ogbn-mag')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        wandb.run.name = f"run-{run + 1}-hidden-{args.hidden_channels}-lr-{args.lr}"
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data.x_dict, data.adj_t_dict,
                         data.y_dict['paper'], train_idx, optimizer)
            result = test(model, data.x_dict, data.adj_t_dict,
                          data.y_dict['paper'], split_idx, evaluator)
            logger.add_result(run, result)
            
            
            train_acc, valid_acc, test_acc = result
            wandb.log({
                'epoch': epoch,
                'loss': loss,
                'train_acc': 100 * train_acc,
                'valid_acc': 100 * valid_acc,
                'test_acc': 100 * test_acc,
            })

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
        if args.save_model:
            model_path = os.path.join(os.getcwd(), f'model_run_{run+1}.pth')
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
            print(f"Model for run {run+1} saved to {model_path}")
            #f_log.write(f"Model for run {run+1} saved to {model_path}\n")

    logger.print_statistics()
    wandb.finish()



if __name__ == "__main__":
    main()