from dgl.nn.pytorch import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import pytorch_forecasting.metrics.MAPE as MAPE


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, mask


def get_locations(source=None):

    fpath_kai = os.path.join('data', 'kaiterra', 'kaiterra_locations.csv')
    fpath_gov = os.path.join('data', 'govdata', 'govdata_locations.csv')
    
    if source == 'combined' or source is None:
        locs_df_kai = pd.read_csv(fpath_kai, usecols=[0,2,3,4], index_col=[0])
        locs_df_gov = pd.read_csv(fpath_gov, index_col=[0])
        locs_df = pd.concat([locs_df_kai, locs_df_gov], axis=0, sort=False)
    elif source == 'kaiterra':
        locs_df = pd.read_csv(fpath_kai, usecols=[0,2,3,4], index_col=[0])
    elif source == 'govdata':
        locs_df = pd.read_csv(fpath_gov, index_col=[0])

    locs_df.sort_index(inplace=True)

    return locs_df


def load_pollution_graph(split='train'):
	locs_df = get_locations("kaiterra")
	data = pd.read_csv('splines.csv')
	if split == 'train':
		data = data[data.timestamp_round.apply(lambda x: datetime.strptime(x, "%Y-%m-%d 00:00:00+05:30") < datetime.strptime('2019-10-01', "%Y-%m-%d"))]
	else:
		data = data[data.timestamp_round.apply(lambda x: datetime.strptime(x, "%Y-%m-%d 00:00:00+05:30") >= datetime.strptime('2019-10-01', "%Y-%m-%d"))]
	features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
	num_nodes = len(locs_df)
	srcs = []
	dsts = []
	for i in range(num_nodes):
		srcs.extend([i]*num_nodes)
		dsts.extend(list(range(num_nodes)))
	# Fully connected graph
	g = DGLGraph((torch.tensor(srcs), torch.tensor(dsts)))
	return g, features, labels, mask


import time
import numpy as np

g, features, labels, mask = load_pollution_graph()

# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=features.size()[1],
          hidden_dim=8,
          out_dim=7,
          num_heads=2)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
for epoch in range(30):
    if epoch >= 3:
        t0 = time.time()

    logits = net(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), np.mean(dur)))

g_t, features_t, labels_t, mask_t = load_pollution_graph("test")

preds = net(features_t)
rmse = nn.MSELoss(preds, labels_t)
mape_f = MAPE()
mape = mape_f.loss(preds, labels_t)
print ("test error RMSE: {:05d}, MAPE: {:05d}".format(rmse, mape))
