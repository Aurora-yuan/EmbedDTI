#coding=utf-8
import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from scipy.sparse import csr_matrix
from utils import *
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import AllChem


def read_data(filename):
    df = pd.read_csv('data/'+filename)
    drugs,proteins,Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])

    return drugs,proteins,Y


# read files(train and test)
davis_train_drugs,davis_train_proteins,davis_train_Y = read_data("davis_train.csv")
davis_test_drugs,davis_test_proteins,davis_test_Y = read_data("davis_test.csv")
drugs = davis_train_drugs+davis_test_drugs
print(len(drugs)) # 30056


# 处理蛋白质序列，构建蛋白质字典
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
# print(seq_dict)


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  

train_prots = [seq_cat(t) for t in davis_train_proteins]
test_prots = [seq_cat(t) for t in davis_test_proteins]
print(len(train_prots))  # 25046
print(len(test_prots))   # 5010


# 处理SMILES序列，转化为原子的图结构

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + # 返回原子上隐式Hs的数量
                    one_of_k_encoding_unk(atom.GetTotalValence(),[0,1,2,3,4,5,6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(),[0,1,2,3,4,5,6,7,8,9,10]) +
                    [atom.GetIsAromatic()] +
                    [atom.IsInRing()]
                    )

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()  # 原子数
    
    features = [] 
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    # 构建有向图
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


compounds = []
for smile in drugs:
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=True)
    compounds.append(lg)
print("原始SMILES数量：", len(compounds))  # 30056
smile_graph = {}
compound_smiles = set(compounds)
print("不重复的SMILES数量:", len(compound_smiles))   # 68
for smile in compound_smiles:
    g = smile_to_graph(smile) # c_size, features, edge_index
    smile_graph[smile] = g
print("simile_graph长度:", len(smile_graph))  # 68



# # 处理SMILES序列，转化为基团的图结构

MST_MAX_WEIGHT = 100 

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

# 构建邻接矩阵和特征矩阵
def cluster_graph(mol,idx):
	n_atoms = mol.GetNumAtoms()
	# if n_atoms == 1: #special case
 #    	return [[0]], []
	cliques = []
	for bond in mol.GetBonds():
		a1 = bond.GetBeginAtom().GetIdx()
		a2 = bond.GetEndAtom().GetIdx()
		if not bond.IsInRing():
			cliques.append([a1,a2])

	ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
	cliques.extend(ssr)

	# nei_list为原子属于哪个基团
	nei_list = [[] for i in range(n_atoms)]
	for i in range(len(cliques)):
	    for atom in cliques[i]:
	        nei_list[atom].append(i)

	#Merge Rings with intersection > 2 atoms
	for i in range(len(cliques)):
		if len(cliques[i]) <= 2: continue
		for atom in cliques[i]:
			for j in nei_list[atom]:
				if i >= j or len(cliques[j]) <= 2: 
					continue
				inter = set(cliques[i]) & set(cliques[j])
				if len(inter) > 2:
					cliques[i].extend(cliques[j])
					cliques[i] = list(set(cliques[i]))
					cliques[j] = []
    
	cliques = [c for c in cliques if len(c) > 0]
	nei_list = [[] for i in range(n_atoms)]
	for i in range(len(cliques)):
		for atom in cliques[i]:
			nei_list[atom].append(i)

	# Build edges and add singleton cliques
	edges = defaultdict(int)
	for atom in range(n_atoms):
		if len(nei_list[atom]) <= 1: 
			continue
		cnei = nei_list[atom]
		bonds = [c for c in cnei if len(cliques[c]) == 2]
		rings = [c for c in cnei if len(cliques[c]) > 4]
		
		for i in range(len(cnei)):
			for j in range(i + 1, len(cnei)):
				c1,c2 = cnei[i],cnei[j]
				inter = set(cliques[c1]) & set(cliques[c2])
				if edges[(c1,c2)] < len(inter):
					edges[(c1,c2)] = len(inter) # cnei[i] < cnei[j] by construction
					edges[(c2,c1)] = len(inter)

	edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
	row,col,data = zip(*edges)
	data = list(data)
	for i in range(len(data)):
		data[i] = 1
	data = tuple(data)
	n_clique = len(cliques)
	clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
	edges = [[row[i],col[i]] for i in range(len(row))]
	
	return cliques, edges



def clique_features(clique,edges,clique_idx,smile):
    NumAtoms = len(clique) # 基团中除去氢原子的原子数
    NumEdges = 0  # 与基团所连的边数
    for edge in edges:
        if clique_idx == edge[0] or clique_idx == edge[1]:
            NumEdges += 1
    mol = Chem.MolFromSmiles(smile)
    NumHs = 0 # 基团中氢原子的个数
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in clique:
            NumHs += atom.GetTotalNumHs()
    # 基团中是否包含环
    IsRing = 0
    if len(clique) > 2:
        IsRing = 1
    # 基团中是否有键
    IsBond = 0
    if len(clique) == 2:
        IsBond = 1

    return np.array(one_of_k_encoding_unk(NumAtoms,[0,1,2,3,4,5,6,7,8,9,10]) + 
        one_of_k_encoding_unk(NumEdges,[0,1,2,3,4,5,6,7,8,9,10]) + 
        one_of_k_encoding_unk(NumHs,[0,1,2,3,4,5,6,7,8,9,10]) + 
        [IsRing] + 
        [IsBond]
        )


cluster = [] # 邻接矩阵, 三维矩阵，每一张图是一个二维矩阵
edges = []  # 连接边矩阵, 三维矩阵，每一张图是一个二维矩阵

for i,smile in enumerate(compound_smiles):
    try:
        mol = get_mol(smile)
        clique,edge = cluster_graph(mol,i)
        cluster.append(clique)
        edges.append(edge)
    except:
        print('Error:',i)

print(len(cluster))  # 68
print(len(edges))   # 68

clique_graph = {}
for i,smile in enumerate(compound_smiles):
    c_features = []
    for idx in range(len(cluster[i])):
        cq_features = clique_features(cluster[i][idx],edges[i],idx,smile)
        c_features.append( cq_features / sum(cq_features) )
    clique_size = len(cluster[i])
    graph = (clique_size, c_features, edges[i])
    clique_graph[smile] = graph

print("clique_graph长度:", len(clique_graph))  # 68

train_drugs, train_prots,  train_Y = np.asarray(davis_train_drugs), np.asarray(train_prots), np.asarray(davis_train_Y)
test_drugs, test_prots,  test_Y = np.asarray(davis_test_drugs), np.asarray(test_prots), np.asarray(davis_test_Y)

train_data = TestbedDataset(root='data', dataset='davis_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph,clique_graph=clique_graph)
test_data = TestbedDataset(root='data', dataset='davis_test', xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph,clique_graph=clique_graph)






