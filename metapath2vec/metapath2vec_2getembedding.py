# %%
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import MetaPath2Vec
from torch_sparse import transpose
import torch
import wandb
import os 
wandb.login(key="API_KEY")

# %%
run = wandb.init(
    # Set the project where this run will be logged
    project="meta-path2vec",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 4,
        "embedding_dim": 128,
        "walk_length": 64,
        "context_size": 7,
        "walks_per_node": 5,
        "num_negative_samples": 5,
        "batch_size": 128,
        "num_workers": 6,
    },
)
 
# %%
dataset = PygNodePropPredDataset(name="ogbn-mag")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

data = dataset[0]

# %%

data.edge_index_dict[('institution', 'employs', 'author')] = transpose(
    data.edge_index_dict[('author', 'affiliated_with', 'institution')],
    None, m=data.num_nodes_dict['author'],
    n=data.num_nodes_dict['institution'])[0]

data.edge_index_dict[('paper', 'written_by', 'author')] = transpose(
    data.edge_index_dict[('author', 'writes', 'paper')], None,
    m=data.num_nodes_dict['author'], n=data.num_nodes_dict['paper'])[0]

data.edge_index_dict[('field_of_study', 'contains', 'paper')] = transpose(
    data.edge_index_dict[('paper', 'has_topic', 'field_of_study')], None,
    m=data.num_nodes_dict['paper'],
    n=data.num_nodes_dict['field_of_study'])[0]

metapath = [
    ('author', 'writes', 'paper'),
    ('paper', 'has_topic', 'field_of_study'),
    ('field_of_study', 'contains', 'paper'),
    ('paper', 'written_by', 'author'),
    ('author', 'affiliated_with', 'institution'),
    ('institution', 'employs', 'author'),
    ('author', 'writes', 'paper'),
    ('paper', 'cites', 'paper'),
    ('paper', 'written_by', 'author'),
]

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

config = wandb.config

model = MetaPath2Vec(data.edge_index_dict, embedding_dim=config.embedding_dim,
                     metapath=metapath, walk_length=config.walk_length, context_size=config.context_size,
                     walks_per_node=config.walks_per_node, num_negative_samples=config.num_negative_samples,
                     sparse=True).to(device)

loader = model.loader(batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config.learning_rate)


def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            avg_loss = total_loss / log_steps
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, Loss: {avg_loss:.4f}')
            wandb.log({"epoch": epoch, "step": i + 1, "loss": avg_loss})
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, Acc: {acc:.4f}')
            wandb.log({"epoch": epoch, "step": i + 1, "accuracy": acc})


@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('paper', batch=torch.arange(0, 736389).to(device))
    y = data.y_dict['paper'].ravel().to(device)
    
    dataset = PygNodePropPredDataset(name='ogbn-mag') 
    split_idx = dataset.get_idx_split()
    #perm = torch.randperm(z.size(0))
    #train_perm = perm[:int(z.size(0) * train_ratio)]
    #test_perm = perm[int(z.size(0) * train_ratio):]
    train_idx = split_idx['train']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)
    
    
    print(f"z device: {z.device}")
    print(f"train_idx device: {train_idx.device}, test_idx device: {test_idx.device}")
    print(f"y device: {y.device}")


    return model.test(z[train_idx], y[train_idx], z[test_idx], y[test_idx], max_iter=150)

   # return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm], max_iter=150)


# Generer en unik identifier baseret på hyperparametrene
hyperparam_identifier = f"lr_{wandb.config.learning_rate}_dim_{wandb.config.embedding_dim}_walklen_{wandb.config.walk_length}_walks_{wandb.config.walks_per_node}"

# Opret en mappe til embeddings baseret på hyperparametrene
output_dir = f"embeddings/{hyperparam_identifier}"
os.makedirs(output_dir, exist_ok=True)

@torch.no_grad()
def save_embeddings():
    model.eval()

    for node_type in data.num_nodes_dict.keys():
        print(f"Fetching embeddings for node type: {node_type}")
        embeddings = model(node_type)

        # Gem som PyTorch tensor med hyperparametre i filnavnet
        torch.save(embeddings.cpu(), f"{output_dir}/{node_type}_embeddings_4epochs.pt")

        # Alternativt gem som CSV med hyperparametre i filnavnet
        embeddings_numpy = embeddings.cpu().numpy()
        with open(f"{output_dir}/{node_type}_embeddings.csv", "w") as f:
            for row in embeddings_numpy:
                f.write(",".join(map(str, row)) + "\n")

    print(f"Embeddings gemt i mappen: {output_dir}")




for epoch in range(1, config.epochs + 1):
    train(epoch)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
    wandb.log({"epoch": epoch, "accuracy": acc})
save_embeddings()




wandb.finish()
