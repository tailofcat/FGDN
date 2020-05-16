"""
 
Mean AUC  0.7774
Mean acc  0.7414
AUC  
[0.7820257533519182, 0.7670250896057348, 0.7732643037302536, 0.7899907075534316, 0.7748572945705563]
acc  
[0.7413793103448276, 0.735632183908046, 0.735632183908046, 0.7471264367816092, 0.7471264367816092]
knnk=20 second run
Mean AUC  0.7721
Mean acc  0.7322
AUC  
[0.7634408602150538, 0.7695473251028806, 0.7767157838842427, 0.7966281693880259, 0.7544139121200054]
acc  
[0.735632183908046, 0.735632183908046, 0.7413793103448276, 0.7413793103448276, 0.7068965517241379]
""" 



import warnings
import os
from os.path import join
import numpy as np
import pandas as pd 
import math
import scipy.io as sio 
import pickle
import csv
import os
import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from tqdm import tqdm
np.random.seed(42)
import h5py

from downloader import fetch_abide

# Connectomes per measure
from connectome_matrices import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
measures = ['correlation', 'partial correlation', 'tangent', 'covariance', 'precision']
atlases = ['AAL', 'HarvardOxford', 'BASC/networks', 'BASC/regions',
               'Power', 'MODL/64', 'MODL/128']

knn_k = 20
nfold = 5
measure = measures[2]
atlas = atlases[-1]
if measure=='correlation':
    name = 'cr'
elif measure=='covariance':
    name = 'co'
elif measure =='precision':
    name = 'pr'
elif measure=='tangent':
    name = 'tg'
elif measure=='precision':
    name = 'pc'
else:
    name = ''
    assert 0

print('Training Knn k: ' + str(knn_k))
print('Training measure: ' + measure)
print('Training measure: ' + atlas)
print('Method name is ' + name)

dimensions = {'AAL': 116,
              'HarvardOxford': 118,
              'BASC/networks': 122,
              'BASC/regions': 122,
              'Power': 264,
              'MODL/64': 64,
              'MODL/128': 128}

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

def _get_paths(phenotypic, atlas, timeseries_dir):
    """
    """
    timeseries = []
    IDs_subject = []
    diagnosis = []
    subject_ids = phenotypic['SUB_ID']
    for index, subject_id in enumerate(subject_ids):
        this_pheno = phenotypic[phenotypic['SUB_ID'] == subject_id]
        this_timeseries = join(timeseries_dir, atlas,
                               str(subject_id) + '_timeseries.txt')
        if os.path.exists(this_timeseries):
            timeseries.append(np.loadtxt(this_timeseries))
            IDs_subject.append(subject_id)
            diagnosis.append(this_pheno['DX_GROUP'].values[0])
    return timeseries, diagnosis, IDs_subject

def to_categorical(y, num_classes=None, dtype='float32'): 
    #将输入y向量转换为数组
#    y = np.int16(y)
    y = np.array(y, dtype='int16')
    #获取数组的行列大小
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    #y变为1维数组
    y = y.ravel()
    #如果用户没有输入分类个数，则自行计算分类个数
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0] 
    #生成全为0的n行num_classes列的值全为0的矩阵
    categorical = np.zeros((n, num_classes), dtype=dtype)
    #np.arange(n)得到每个行的位置值，y里边则是每个列的位置值
    categorical[np.arange(n), y] = 1
    #进行reshape矫正
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
# Path to data directory where timeseries are downloaded. If not
# provided this script will automatically download timeseries in the
# current directory.
'''
Read data
''' 

print('Cross Validation')
all_max_acc = []
all_max_AUC = []
all_last_acc = []
all_last_AUC = []
for cv_n in range(nfold):
    if os.path.exists('./processed/Train_K'+str(knn_k)+'_M'+name+'CV'+str(nfold)+str(cv_n)+'.dataset'):
        pass
    else:
        timeseries_dir = None
        # If provided, then the directory should contain folders of each atlas name
        if timeseries_dir is not None:
            if not os.path.exists(timeseries_dir):
                warnings.warn('The timeseries data directory you provided, could '
                              'not be located. Downloading in current directory.',
                              stacklevel=2)
                timeseries_dir = fetch_abide(data_dir='./ABIDE')
        else:
            # Checks if there is such folder in current directory. Otherwise,
            # downloads in current directory
            timeseries_dir = './ABIDE'
            if not os.path.exists(timeseries_dir):
                timeseries_dir = fetch_abide(data_dir='./ABIDE')
        
        # Path to data directory where predictions results should be saved.
        predictions_dir = None
        
        if predictions_dir is not None:
            if not os.path.exists(predictions_dir):
                os.makedirs(predictions_dir)
        else:
            predictions_dir = './ABIDE/predictions'
            if not os.path.exists(predictions_dir):
                os.makedirs(predictions_dir)
        
        
        # prepare dictionary for saving results
        columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
                   'dataset', 'covariance_estimator', 'dimensionality']
        results = dict()
        for column_name in columns:
            results.setdefault(column_name, [])
        
        pheno_dir = 'Phenotypic_V1_0b_preprocessed1.csv'
        phenotypic = pd.read_csv(pheno_dir)
        
#        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=1)
        cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=1)
        
        #    atlas = atlases[0]#取第一个脑模板
        timeseries, diagnosis, _ = _get_paths(phenotypic, atlas, timeseries_dir)
        
        _, classes = np.unique(diagnosis, return_inverse=True)
        iter_for_prediction = cv.split(timeseries, classes)
        print('Loading cross-valid data')
        train_index, test_index = next(iter_for_prediction)
        #由上一句替代#for index, (train_index, test_index) in enumerate(iter_for_prediction):
    #    measure = measures[2]
        
        connections = ConnectivityMeasure(
                        cov_estimator=LedoitWolf(assume_centered=True),
                        kind=measure)
        
        conn_coefs = connections.fit_transform(timeseries, vectorize=False)#vectorize=True取矩阵上三角, False=全矩阵
        
        
        def calculate_edge(conn_coefs):
            meanA = np.mean(conn_coefs, axis=0)
            
            edge_index0 = []
            edge_index1 = np.repeat(np.arange(0,np.shape(meanA)[0]),knn_k)
            for vi in range(np.shape(meanA)[0]):
                vec = meanA[vi,:]
                minx = np.argsort(vec)
                chose_minx = minx[-knn_k-1:-1]
                edge_index0.extend(chose_minx)
            
            edge_index0 = np.array(edge_index0)  
            edge_index = np.reshape(np.concatenate((edge_index0, edge_index1)), (2,-1))
            return edge_index
        
#        is_c1 = classes==0
#        conn_coefs_c1 = conn_coefs[is_c1]
#        conn_coefs_c2 = conn_coefs[~is_c1]
#        edge_index_c1 = calculate_edge(conn_coefs_c1)
#        edge_index_c2 = calculate_edge(conn_coefs_c2)
        
        train_conn_coefs = conn_coefs[train_index,:] 
        is_c1 = classes[train_index]==0 
        conn_coefs_c1 = train_conn_coefs[is_c1]
        conn_coefs_c2 = train_conn_coefs[~is_c1]
        edge_index_c1 = calculate_edge(conn_coefs_c1) # 对比c1p和c1中0节点与其他节点的连接情况，当前，10连接4个类似
        edge_index_c2 = calculate_edge(conn_coefs_c2)
    
        classes2 = to_categorical(classes, 2)
        X = conn_coefs[train_index,:]
        Y = classes2[train_index,:]
        testX = conn_coefs[test_index,:]
        testY = classes2[test_index,:]
        print(np.shape(X))
        print(np.shape(Y))
        print(np.shape(testX))
        print(np.shape(testY))

    class ABIDEDatasetTrain(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(ABIDEDatasetTrain, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
        @property
        def raw_file_names(self):
            return []
        @property
        def processed_file_names(self):
            return ['./Train_K'+str(knn_k)+'_M'+name+'CV'+str(nfold)+str(cv_n)+'.dataset']
    
        def download(self):
            pass
        
        def process(self):
            #eimatrix = [np.random.randint(0,64,(200),dtype=np.int64), np.random.randint(0,64,(200),dtype=np.int64)]
            data_list = []
#            X, Y, testX, testY, edge_c1, edge_c2 = preprocess_graph(cv_n) 
            # process by samples
            for sample_id in tqdm(range(np.shape(Y)[0])): 
                x = torch.FloatTensor(X[sample_id,:])#.unsqueeze(1)  
    
                y = torch.FloatTensor(Y[sample_id,:])
                
                edge_index1 = torch.LongTensor(edge_index_c1)  
                edge_index2 = torch.LongTensor(edge_index_c2)  
                
                data = Data(x=x, y=y, edge_index1=edge_index1, edge_index2=edge_index2)#Data(x=x, pos=pos, y=y)
    
                data_list.append(data) 
    
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            
    class ABIDEDatasetTest(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(ABIDEDatasetTest, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
        @property
        def raw_file_names(self):
            return []
        @property
        def processed_file_names(self):
            return ['./Test_K'+str(knn_k)+'_M'+name+'CV'+str(nfold)+str(cv_n)+'.dataset']
    
        def download(self):
            pass
        
        def process(self):
            #eimatrix = [np.random.randint(0,64,(200),dtype=np.int64), np.random.randint(0,64,(200),dtype=np.int64)]
            data_list = []
#            X, Y, testX, testY, edge_c1, edge_c2 = preprocess_graph(cv_n)
            # process by samples
            for sample_id in tqdm(range(np.shape(testY)[0])):
                
                node_features = torch.FloatTensor(testX[sample_id,:])#.unsqueeze(1) 
    
                x = node_features
                    
                y = torch.FloatTensor(testY[sample_id,:])
    
                edge_index1 = torch.LongTensor(edge_index_c1)  
                edge_index2 = torch.LongTensor(edge_index_c2)  
                
                data = Data(x=x, y=y, edge_index1=edge_index1, edge_index2=edge_index2)#Data(x=x, pos=pos, y=y)
    
                data_list.append(data) 
    
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
    dataset = ABIDEDatasetTrain('./')
    test_dataset = ABIDEDatasetTest('./') 

    from torch.nn import Sequential as Seq, Linear, ReLU
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import remove_self_loops, add_self_loops
    
    dataset = dataset.shuffle()
    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 10]
    #val_dataset = dataset[one_tenth_length*8:one_tenth_length * 10]
    
    
    #len(train_dataset), len(val_dataset), len(test_dataset)
    
    from torch_geometric.data import DataLoader
    batch_size= 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True )
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
    
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    from torch_geometric.nn import GraphConv, FeaStConv, GATConv, TAGConv, ChebConv, GINConv, ARMAConv
    from torch_geometric.nn import GatedGraphConv, SAGEConv, SGConv, GCNConv, AGNNConv
    from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
    from torch.nn import PReLU
    from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
    import torch.nn.functional as F
    from torch.nn import Linear, Dropout, Dropout2d, Conv1d, AvgPool1d, MaxPool1d, LayerNorm, BatchNorm1d
    
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.channels = 128
            self.timestep = 128
            self.poolrate = 1
            
            self.gconv1 = ChebConv(self.timestep, 64, K=3)
            self.activ1 = PReLU() 
            self.drop1 = Dropout(0.1)
            self.gconv2 = ChebConv(64, 64, K=3)
            self.activ2 = PReLU()
            self.drop2 = Dropout(0.1) 
            
            self.linend = Linear(64* self.channels , 1) 
            
        def forward(self, data, edge_index1, edge_index2, batch):
    #        x, edge_index, batch = data.x, data.edge_index, data.batch  
            
            x = data.reshape(-1, 128) 
    
            x1 = self.gconv1(x, edge_index1) 
            x1 = self.activ1(x1)
            x1 = self.drop1(x1)
            x1 = self.gconv2(x1, edge_index1) 
            x1 = self.activ2(x1)
            x1 = self.drop2(x1) 
            x1 = x1.reshape(-1, 64*self.channels )
            x1 = self.linend(x1)
            x1 = torch.sigmoid(x1)
            
            x2 = self.gconv1(x, edge_index2) 
            x2 = self.activ1(x2)
            x2 = self.drop1(x2)
            x2 = self.gconv2(x2, edge_index2) 
            x2 = self.activ2(x2)
            x2 = self.drop2(x2) 
            x2 = x2.reshape(-1, 64*self.channels)
            x2 = self.linend(x2)
            x2 = torch.sigmoid(x2)
            
            x =  torch.cat([x1,x2], dim=1) 
#            x = x.view(-1,2)
#            if self.training:   
#                x = torch.cat([x1[label[:,0]==0,:],x2[label[:,0]==1,:]], dim=0) 
#            else: 
#                x = torch.cat([x1,x2], dim=1) 
#                x = x.view(-1,1,2) 
#                x = self.poolend(x) 
#                x = x.view(-1,1) 
            
            return x
    class FocalLoss(torch.nn.Module):
    
        def __init__(self, focusing_param=2, balance_param=0.25):
            super(FocalLoss, self).__init__()
    
            self.focusing_param = focusing_param
            self.balance_param = balance_param
    
        def forward(self, output, target):  
            target = torch.argmax(target, axis=-1)
            cross_entropy = F.cross_entropy(output, target)
            cross_entropy_log = torch.log(cross_entropy)
            logpt = - F.cross_entropy(output, target)
            pt    = torch.exp(logpt)
    
            focal_loss = -((1 - pt) ** self.focusing_param) * logpt
    
            balanced_focal_loss = self.balance_param * focal_loss
    
            return balanced_focal_loss
    
    device = torch.device('cuda')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    #    RMSprop
    crit = torch.nn.BCELoss()#BCEWithLogitsLoss(pos_weight=torch.Tensor([39, 1]).to(device))
    '''L1Loss BCELoss
    '''
    
        
    def train():
        model.train()
    
        loss_all = 0
        for data in train_loader: 
            
            
            data = data.to(device)
            optimizer.zero_grad()
            label = data.y.view(-1,2).to(device) 
            output = model(data.x, data.edge_index1, data.edge_index2, data.batch)
            
            loss = crit(output, label)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step() 
    
        return loss_all / len(train_dataset)
    
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    def evaluate(loader, save_result=False):
        model.eval()
    
        predictions = []
        labels = [] 
        with torch.no_grad():
            for data in loader:
                label = data.y.view(-1,2)
    #            label = data.y.detach().cpu().numpy() 
    #            label = np.reshape(label, (np.shape(pred)[0],2)) 
                data = data.to(device)
                pred = model(data.x, data.edge_index1, data.edge_index2, data.batch).detach().cpu().numpy() 
                
                pred = pred 
                label = label  
                pred = np.squeeze(pred)
                predictions.append(pred)
                labels.append(label)
    
        predictions = np.vstack(predictions)
        labels = np.vstack(labels)
        
        AUC = roc_auc_score(labels[:,0], predictions[:,0]) 
        
        predictions = np.argmax(predictions, axis = -1)
        labels = np.argmax(labels, axis = -1)
        acc = accuracy_score(labels, predictions)
    #    acc = 0
        
        return AUC, acc
    
    
    import time
    max_test_acc = 0
    max_test_AUC = 0
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=30, gamma=0.9)
    for epoch in range(500):
        t0 = time.time()
        loss = train()
        train_AUC, train_acc = evaluate(train_loader)
    #    val_acc = evaluate(val_loader)    
        test_AUC, test_acc = evaluate(test_loader)
        if max_test_acc < test_acc:
            max_test_acc = test_acc
        
        if max_test_AUC < test_AUC:
            max_test_AUC = test_AUC
        t1 = time.time()
        print('CV:{:01d}Epoch: {:03d}, Loss:{:.3f}, Trainacc:{:.3f}, TestAUC:{:.2f}, Testacc:{:.2f}, MaxAUC: {:.4f}, Maxacc: {:.4f}, Time: {:.2f}'.
                  format(cv_n, epoch+1, loss, train_acc, test_AUC, test_acc, max_test_AUC, max_test_acc, (t1-t0)))
    
        scheduler.step()
        if train_acc>0.99:
            break
    print('Results::::::::::::')
    print('CV:{:01d}Epoch: {:03d}, Loss:{:.3f}, Trainacc:{:.3f}, TestAUC:{:.2f}, Testacc:{:.2f}, MaxAUC: {:.4f}, Maxacc: {:.4f}, Time: {:.2f}'.
                  format(cv_n, epoch+1, loss, train_acc, test_AUC, test_acc, max_test_AUC, max_test_acc, (t1-t0)))
    
    all_max_acc.append(max_test_acc)
    all_max_AUC.append(max_test_AUC)
    all_last_acc.append(test_acc)
    all_last_AUC.append(test_AUC)
print(np.mean(all_max_AUC))
print(all_max_AUC)

print(np.mean(all_max_acc))
print(all_max_acc)

print(np.mean(all_last_AUC))
print(all_last_AUC)

print(np.mean(all_last_acc))
print(all_last_acc)

FilePath='./result'  
if os.path.exists(FilePath):   ##目录存在，返回为真
    print( 'dir exists'  ) 
else:
    print( 'dir not exists')
    os.makedirs(FilePath) 
    #os.mkdir(FilePath) 
with open('./result/gnn_'+'K'+str(knn_k)+'_M'+name+'CV'+str(nfold)+'.txt', 'w') as f:
    f.writelines('Result saved, knn = '+ str(knn_k))
    f.writelines('\nMean AUC  '+str(round(np.mean(all_max_AUC), 4)))
    f.writelines('\nMean acc  '+str(round(np.mean(all_max_acc), 4)))
    f.writelines('\nAUC  \n'+str(all_max_AUC))
    f.writelines('\nacc  \n'+str(all_max_acc))
    
    f.writelines('\nLastMean AUC  '+str(round(np.mean(all_last_AUC), 4)))
    f.writelines('\nLastMean acc  '+str(round(np.mean(all_last_acc), 4)))
    f.writelines('\nLastAUC  \n'+str(all_last_AUC))
    f.writelines('\nLastacc  \n'+str(all_last_acc))
    
    