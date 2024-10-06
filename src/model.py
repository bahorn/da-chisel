"""
mostly stolen example code with some patches
"""
import random
import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from consts import BATCH_SIZE, EPOCHES, LR, HIDDEN


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class GCN(Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # single output
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCNScorer:
    def __init__(self, path):

        self._device = get_device()
        self._model = GCN(hidden_channels=HIDDEN)
        self._model.load_state_dict(torch.load(path, weights_only=True))
        self._model = self._model.to(self._device)
        self._model.eval()

    def score(self, p, subsets):
        to_compute = [
            from_networkx(
                p.simplify(subset),
                group_node_attrs=['threshold', 'degree']
            ) for subset in subsets
        ]
        # now make it a batch
        dl = DataLoader(to_compute, batch_size=BATCH_SIZE)
        res = []
        for data in dl:
            data = data.to(self._device)
            out = self._model(data.x, data.edge_index, data.batch)
            res += list(out.detach().to('cpu').flatten().tolist())
        return res


def train(dataset_raw):
    device = get_device()

    dataset = []
    for graph, score in dataset_raw:
        data = from_networkx(graph, group_node_attrs=['threshold', 'degree'])
        data.y = torch.tensor([[score]], dtype=torch.float32)
        dataset.append(data)

    train_count = int(len(dataset) * 0.8)
    test_count = len(dataset) - train_count

    random.shuffle(dataset)

    train_dataset = dataset[:test_count]
    test_dataset = dataset[test_count:]

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = GCN(hidden_channels=HIDDEN)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()

        for data in train_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(loader):
        model.eval()

        loss = 0.0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss += criterion(out, data.y)
        return loss

    for epoch in range(0, EPOCHES):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f'Epoch: {epoch:03d}, ',
            f'Train Loss: {train_acc:.4f}, ',
            f'Test Loss: {test_acc:.4f}'
        )

    torch.save(model.state_dict(), './model.pt')
