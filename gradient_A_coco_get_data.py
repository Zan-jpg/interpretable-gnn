#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:40:16 2020

@author: lizan
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
import math
from urllib.request import urlretrieve
import torch
import heapq
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter
import torch.utils.data as tordata
import networkx as nx
from community import community_louvain
import matplotlib as mpl
from torch.autograd import Variable
from sklearn.cluster import SpectralClustering
from tempfile import TemporaryFile
import csv
import json
from networkx.readwrite import json_graph
label_dic ={"airplane": 0, "apple": 1, "backpack": 2, "banana": 3, "baseball bat": 4, "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9, "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14, "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19, "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24, "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29, "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34, "hair drier": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39, "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44, "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49, "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54, "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59, "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64, "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69, "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74, "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79}
urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def download_coco2014(root, phase):
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    data = os.path.join(root, 'data/')
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file,data)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.Popen('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
            
        del img_i
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')
    

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) #- 1
        target[labels] = 1
        return (img, filename, self.inp,labels), target
    
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

A_=gen_A(num_classes=80,t=0.4, adj_file='data/coco/coco_adj.pkl')
A_=torch.from_numpy(A_)
#A=gen_A(num_classes=80,t=0.4, adj_file='/Users/lizan/Desktop/data/coco/coco_adj.pkl')
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        
        
        
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp, A):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        
       # adj = gen_adj(Parameter(A.float()))
        adj = gen_adj(A.float())
        
        
        x = self.gc1(inp, adj)
        
        x = self.relu(x)
        x = self.gc2(x, adj)
        #print(x)
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        #x= F.softmax(x)
        x=torch.sigmoid(x)
        #x[x<=0] = 0
        #x[x>0] = 1
        return x



   

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]


def gcn_resnet101(num_classes, t, pretrained=True, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, in_channel=in_channel)

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)
#use pretrained ML_GCNmodel
        
model = gcn_resnet101(num_classes=80, t=0.4)
checkpoint=torch.load('coco_checkpoint.pth.tar',map_location=torch.device('cpu'))

                     
model.load_state_dict(checkpoint['state_dict'],False)                     
model.eval()






 


 
    
def get_val_result():
    
    global pred_list
    global val_loader
    pred_list1 =[]

    pred_list2=[]

    use_gpu = torch.cuda.is_available()

    print('val starting')
    val_dataset = COCO2014('data/coco', phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
    val_dataset.transform = transforms.Compose([
                Warp(448),
                transforms.ToTensor(),
                normalize,
            ])
    
    #use_dataset,unuse_dataset = train_test_split(val_dataset, test_size=0.01)
    val_loader = tordata.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    for i, (input, target) in enumerate(val_loader):
        
        
        
        if i > 0:
            break
        
        
        #pred1=model(input[0],input[2],torch.mul(A_,torch.FloatTensor(80, 80).uniform_(1, 1)))
        pred1=model(input[0],input[2],torch.eye(80))
        pred2=model(input[0],input[2],A_)
        #print('labels...',input[3])
        pred_list1.append(pred1.data.numpy())
        pred_list2.append(pred2.data.numpy())
     
        
        #print("output....",pred2)
        a=np.where(pred2>0.5)
        #print(a)
        a=list(a[1])
        for i in range(len(a)):
            a[i]=get_key(label_dic,a[i])
        print("predict label.....",a)
        
        
        
        for i, a in enumerate(input[3]):
            input[3][i]=get_key(label_dic,a.numpy())
        print("ground truth label....",input[3])
            
    pred_list1 = np.concatenate(pred_list1)
    
    pred_list2 = np.concatenate(pred_list2)
    
    #use_gpu = torch.cuda.is_available()
    
    #print("get the final result{}".format(pred_list1-pred_list2))
    


    

get_val_result()
   





model.eval()
label_list=[]
for k, v in label_dic.items():
    label_list.append(k)

def make_label_dict(labels):
    l = {}
    for i, label in enumerate(labels):
        l[i] = label
    return l
#print(make_label_dict(label_list))
def save(G, fname):
       json.dump(dict(nodes=[[n, G.node[n]] for n in G.nodes()],
                      edges=[[u, v, G.edge[u][v]] for u,v in G.edges()]),
                 open(fname, 'w'), indent=2)
def show_graph_with_labels(adjacency_matrix, mylabels,graph_number):
    adjacency_matrix_edge_weight_list=[]
    adjacency_matrix_edge_weight=adjacency_matrix.copy()
    adjacency_matrix[np.nonzero(adjacency_matrix)]=1
    rows, cols = np.where(adjacency_matrix ==1)
    
    
    transp=np.transpose(np.nonzero(adjacency_matrix_edge_weight))
    for index in range(len(transp)):
        row,col = transp[index]
        adjacency_matrix_edge_weight_list.append(adjacency_matrix_edge_weight[row,col])
    
    edges = zip(rows.tolist(), cols.tolist(),adjacency_matrix_edge_weight_list)
   
    "add threshold as 0.01 for edge weight"
    estrong = [(u,v,d) for (u,v,d) in edges if abs(d)> 0.01]
    
    
    gr =nx.Graph()
    
    gr.add_weighted_edges_from(estrong)
    
 
    
    gr.to_directed()
    pos = nx.spring_layout(gr)
    
    #......add labels to node......
    #print(len(gr.nodes()))
    for v in gr.nodes():
        gr.node[v]['state']=mylabels[v]
    node_labels = nx.get_node_attributes(gr,'state')
    nx.draw_networkx_labels(gr, pos, labels = node_labels)
    
        

    partition = community_louvain.best_partition(gr,weight='weight')
    for node, cluster in dict(partition).items():
        gr.node[node]['community'] = cluster
    edges = gr.edges()
    weights = [gr[u][v]['weight'] for u,v in edges]
    
    
    #print(gr.edges.data())
    for comm in set(partition.values()):
        key_list=[]
        print("Community %d"%comm)
        for key in partition:
            if partition[key] == comm:
                key_list.append(key)
        for k,a in enumerate(key_list):
            key_list[k]=get_key(label_dic,a)
            
        print(key_list)
    
    
    nx.draw(gr,pos, node_size=50,cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(gr, pos, arrowstyle='->',arrowsize=10,edge_cmap=plt.cm.Greys, width=weights)
    
    
    #plt.colorbar(edges)
    json_data=json_graph.node_link_data(gr)
    del json_data['directed']
    del json_data['multigraph']
    del json_data['graph']
    
    
    
    #print(json_data)
    for u in range(len(gr.nodes())):
        json_data['nodes'][u].pop('id')
        json_data['nodes'][u]["id"] = json_data['nodes'][u].pop('state')
        json_data['nodes'][u]["group"] = json_data['nodes'][u].pop('community')
        
    for v in range(len(gr.edges())):
         
         json_data['links'][v]["value"] = json_data['links'][v].pop('weight')
         json_data['links'][v]["value"]=_A_[json_data['links'][v]['source']][json_data['links'][v]['target']]*saliency[json_data['links'][v]['source']][json_data['links'][v]['target']].numpy()
         json_data['links'][v]['source']="".join(get_key(label_dic,json_data['links'][v]['source']))
         json_data['links'][v]['target']="".join(get_key(label_dic,json_data['links'][v]['target']))
         
    print(json_data)
    j = json.dumps(json_data)
    f2 = open('graph_json{}.json'.format(graph_number), 'w')
    f2.write(j)
    f2.close()
    
    plt.title("best_partition")
    
    plt.gcf().set_size_inches(15, 10)
    
    plt.show()
    
   

    
"use spectual clustering method"   

def show_graph_with_labels_spectual_clustering(adjacency_matrix, mylabels):
    adjacency_matrix_edge_weight_list=[]
    adjacency_matrix_edge_weight=adjacency_matrix.copy()
    adjacency_matrix[np.nonzero(adjacency_matrix)]=1
    rows, cols = np.where(adjacency_matrix ==1)
    
    
    transp=np.transpose(np.nonzero(adjacency_matrix_edge_weight))
    for index in range(len(transp)):
        row,col = transp[index]
        adjacency_matrix_edge_weight_list.append(adjacency_matrix_edge_weight[row,col])
    
    edges = zip(rows.tolist(), cols.tolist(),adjacency_matrix_edge_weight_list)
    
    "add threshold as 0.01 for edge weight"
    estrong = [(u,v,d) for (u,v,d) in edges if d > 0.02]
    
    
    gr =nx.Graph()
    
    gr.add_weighted_edges_from(estrong)
    adjacency_matrix_edge_weight[adjacency_matrix_edge_weight<0.02]=0
    sc = SpectralClustering(6, affinity='precomputed', n_init=100,
                         assign_labels='discretize')
    sc.fit_predict(adjacency_matrix_edge_weight) 
    print(sc.labels_)
 
    
    gr.to_directed()
    pos = nx.spring_layout(gr)
    #edge_labels = dict( ((u, v), d["weight"]) for u, v, d in gr.edges(data=True) )
    #print(edge_labels)
    #......add labels to node......
    print(len(gr.nodes()))
    
    for v in gr.nodes():
        gr.node[v]['state']=mylabels[v]
    node_labels = nx.get_node_attributes(gr,'state')
    nx.draw_networkx_labels(gr, pos, labels = node_labels)
    
    
    edges = gr.edges()
    weights = [gr[u][v]['weight'] for u,v in edges]
    #print(gr.edges.data())
    for comm in range(0,6):
        key_list=[]
        print("Community %d"%comm)
        for key in gr.nodes():
            if sc.labels_[key] == comm:
                key_list.append(key)
        for k,a in enumerate(key_list):
            key_list[k]=get_key(label_dic,a)
            
        print(key_list)
    node_list=[]
    for key in gr.nodes():
            node_list.append(sc.labels_[key])
    #M = gr.number_of_edges()
    #edge_colors = range(2, M + 2)
    
    
    nx.draw(gr,pos, node_size=50,cmap=plt.cm.RdYlBu, node_color=node_list)
    nx.draw_networkx_edges(gr, pos, arrowstyle='->',arrowsize=10,edge_cmap=plt.cm.Greys, width=weights)
    plt.title("spectual clustering")
    plt.gcf().set_size_inches(15, 10)
    plt.show()
    


   
    


      
def largest_indices(ary, n):
   """Returns the n largest indices from a numpy array."""
   flat = ary.flatten()
   indices = np.argpartition(flat, -n)[-n:]
   indices = indices[np.argsort(-flat[indices])]
   return np.unravel_index(indices, ary.shape)

for i, (input, target) in enumerate(val_loader):
    if i >0:
        break
    print("compute saliency maps start...")
    A=Variable(A_, requires_grad=True)
    
    photo=Variable(input[0], requires_grad=True)
    feature=Variable(input[2], requires_grad=True)
    pred=model(photo,feature,A)
    

    pred.backward(gradient=torch.ones_like(pred),retain_graph=True)
    saliency=A.grad
    
    #print(feature.grad)
    #print(photo.grad)
    
   #print(saliency)
    #saliency=F.relu(saliency)
    abs_saliency=abs(saliency)
    #print(saliency)
    
    abs_saliency = abs_saliency.numpy()
    max_saliency_list=[]
    index=[]
    max_index=largest_indices(abs_saliency,10)
    index1=list(max_index[0])
    index2=list(max_index[1])
    index.append(index1)
    index.append(index2)
    
    for a in range(len(index[0])): 
        max_saliency=saliency[index[0][a]][index[1][a]]
        
        max_saliency_list.append(max_saliency.data.numpy())
    #print(max_saliency_list)
    
    for p in range(len(index)):
      for b in range(len(index[0])): 
        index[p][b]=get_key(label_dic,index[p][b])
      
    
    max_saliency_list=np.array(max_saliency_list)
    index=np.array(index).squeeze()
    max_saliency_list = max_saliency_list[np.newaxis,:]
    print(np.concatenate((index,max_saliency_list)).T)
    
    abs_saliency_=abs_saliency.copy()
    
    abs_saliency_[abs_saliency_<=1] = 0
    abs_saliency_[abs_saliency_>1] = 1
    _A_=A_.numpy().copy()
    _A_[np.nonzero(_A_)]=1
    
    
    
    
    #transp=np.transpose(np.nonzero(A_premium_edge_weight))
    #A_premium[np.nonzero(A_premium.numpy())]=1

    "show node clustering of graph"
   
        
         
    
    #show_graph_with_labels(_A_, make_label_dict(label_list),i+1))
    #show_graph_with_labels(abs_saliency, make_label_dict(label_list),i+1))
    show_graph_with_labels(_A_*abs_saliency, make_label_dict(label_list),i+1)
    #show_graph_with_labels_spectual_clustering(A_premium, make_label_dict(label_list))
    
    
#without threshold
    
    column=abs_saliency.sum(axis=0)
    
    top_column_idx=column.argsort()[::-1][0:5].tolist()
    print('top5 column index.....',top_column_idx)
    for b,a in enumerate(top_column_idx):
        top_column_idx[b]=get_key(label_dic,a)
    print('top5 column.....',top_column_idx)
    
    row=abs_saliency.sum(axis=1)
   
    
    
    top_row_idx=row.argsort()[::-1][0:5].tolist()
    print('top5 row index.....',top_row_idx)
    for p,a in enumerate(top_row_idx):
        top_row_idx[p]=get_key(label_dic,a)
    print('top10 row.....',top_row_idx)
    
#with threshold  
    '''
    column=abs_saliency_.sum(axis=0)
    
    
    top_column_idx=column.argsort()[::-1][0:2].tolist()
    print('top5 column index.....',top_column_idx)
    for t,a in enumerate(top_column_idx):
        top_column_idx[t]=get_key(label_dic,a)
    print('top5 column.....',top_column_idx)
    
    row=abs_saliency_.sum(axis=1)
    
    
    
    top_row_idx=row.argsort()[::-1][0:2].tolist()
    print('top5 row index.....',top_row_idx)
    for p,a in enumerate(top_row_idx):
        top_row_idx[p]=get_key(label_dic,a)
    print('top5 row.....',top_row_idx)
   

    '''
    
    


  
    
    input[0] = input[0].squeeze()
    input[0] = input[0].permute(1,2,0) 
    plt.imshow(input[0])

   
    plt.axis('off')
    plt.title("input image{}".format(i+1))
    plt.show()
    plt.title("abs_saliency map{}".format(i+1))
    
    color_map=plt.imshow(abs_saliency, cmap=plt.cm.hot)
    
    color_map.set_cmap("Blues_r")
    
    plt.colorbar()
    plt.gcf().set_size_inches(12, 5)
    plt.show()
    
    #csvFile_A_premium = open("A_premium map{}.csv".format(i+1), "w")
    #writer = csv.writer(csvFile_A_premium)
    #csvFile_abs_saliency = open("abs_saliency map{}.csv".format(i+1), "w")
    #writer = csv.writer(csvFile_abs_saliency)
    c=[]
    row1=[]
    row2=[]
    row3=[]
    row4=[]
    for u in range(80):
        for v in range(80):
            
            row1.append(get_key(label_dic,u))
            row2.append(get_key(label_dic,v))
            row3.append((_A_*abs_saliency)[u][v])
            row4.append(abs_saliency[u][v])
    row1=np.vstack(row1)
    row2=np.vstack(row2)
    row3=np.vstack(row3)
    row4=np.vstack(row4)
    c=np.hstack((row1,row2,row3)) 
    d=np.hstack((row1,row2,row4))            
    
    #writer.writerows(c)
    #csvFile_A_premium.close()
    #writer.writerows(d)
    #csvFile_abs_saliency.close()
    plt.title("A_premium saliency map{}".format(i+1))
    color_map_=plt.imshow(_A_*abs_saliency, cmap=plt.cm.hot)
    plt.axis('off')
    color_map_.set_cmap("Blues_r")
    plt.colorbar()
    plt.gcf().set_size_inches(12, 5)
    plt.show()
#with threshold   
    '''
    plt.axis('off')
    plt.title("input image{}".format(i+1))
    plt.subplot(1, 2, 2)
    plt.title("saliency map{}".format(i+1))
    color_map=plt.imshow(abs_saliency_, cmap=plt.cm.hot)
    plt.axis('off')
    color_map.set_cmap("Blues_r")
    plt.colorbar()
    plt.gcf().set_size_inches(12, 5)
    plt.show()

    '''    
use_cuda = torch.cuda.is_available()
def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)

print(neighborhoods(_A_,5,use_cuda)


