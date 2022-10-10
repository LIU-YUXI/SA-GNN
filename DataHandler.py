import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import os
import random
def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())
'''
def negSamp_fre(temLabel, sampSize, neg_frequency):
    negset = np.setdiff1d(neg_frequency,temLabel)
    print(temLabel)
    negset = np.concatenate([negset,[neg_frequency[0]]*(args.item-len(negset))],axis=-1)
    return negset

'''

def negSamp_fre(temLabel, sampSize, neg_frequency,pos_los):
    negset = [None] * sampSize
    cur = 0
    i = 0
    # print(temLabel)
    while cur < sampSize:
        rdmItm = neg_frequency[-i]# 
        # rdmItm = np.random.choice(args.item)
        # print(rdmItm,temLabel[rdmItm])
        if rdmItm != pos_los and temLabel[rdmItm] == 0:
            negset[cur] = rdmItm
            cur += 1
        i += 1
    return negset

def negSamp(temLabel, sampSize, nodeNum,trnPos, item_with_pop):
	negset = [None] * sampSize
	cur = 0
	# print(trnPos)
	while cur < sampSize:

		# rdmItm = random.choice(item_with_pop)
		# rdmItm = np.random.choice(sequence[rdmItm],1)
		rdmItm = np.random.choice(nodeNum)
		# if rdmItm not in temLabel and rdmItm != trnPos:
		if temLabel[rdmItm] == 0 and rdmItm not in trnPos:
			negset[cur] = rdmItm
			cur += 1
	return negset

def posSamp(user_sequence,sampleNum):
	indexs=np.random.choice(np.array(range(len(user_sequence))),sampleNum)
	# print(indexs)
	return user_sequence[indexs.sort()]
def transToLsts(mat, mask=False, norm=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.int32)

	if norm:
		rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
		colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
		for i in range(len(data)):
			row = indices[i, 0]
			col = indices[i, 1]
			data[i] = data[i] * rowD[row] * colD[col]

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.int32)
	return indices, data, shape

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = './Datasets/Yelp/'
		elif args.data == 'gowalla':
			predir = './Datasets/gowalla/'
		elif args.data == 'amazon':
			predir = './Datasets/amazon/'
		else:
			predir='./Datasets/'+args.data+'/'
		self.predir = predir
		self.trnfile = predir + 'trn_mat_time'
		self.tstfile = predir + 'tst_int'
		self.trnposfile = predir + 'train_pos'
		self.neg_sequency_file = predir + 'sort'
		self.sequencefile=predir+'sequence'
		self.test_dictfile=predir+'test_dict'
	def LoadData(self):
		if args.percent > 1e-8:
			print('noised')
			with open(self.predir + 'noise_%.2f' % args.percent, 'rb') as fs:
				trnMat = pickle.load(fs)
		else:
			with open(self.trnfile, 'rb') as fs:
				# print(pickle.load(fs))
				trnMat = pickle.load(fs)# (pickle.load(fs) != 0).astype(np.float32)
		# test set
		with open(self.tstfile, 'rb') as fs:
			tstInt = np.array(pickle.load(fs))
		with open(self.trnposfile, 'rb') as fs:
			self.trnPos = np.array(pickle.load(fs))
		with open(self.neg_sequency_file, 'rb') as fs:
			self.neg_sequency = pickle.load(fs)
			print("neg_fre:",len(self.neg_sequency))
		with open(self.sequencefile, 'rb') as fs:
			self.sequence = pickle.load(fs)
		if os.path.isfile(self.test_dictfile):
			with open(self.test_dictfile, 'rb') as fs:
				self.test_dict = pickle.load(fs)
		print("tstInt",tstInt)
		tstStat = (tstInt != None)
		print("tstStat",tstStat,len(tstStat))
		tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
		print("tstUsrs",tstUsrs,len(tstUsrs))
		self.trnMat = trnMat[0]
		self.subMat = trnMat[1]
		self.timeMat = trnMat[2]
		print("trnMat",trnMat[0],trnMat[1],trnMat[2])
		print(self.neg_sequency)
		self.tstInt = tstInt
		self.tstUsrs = tstUsrs
		args.user, args.item = trnMat[0].shape
		self.prepareGlobalData()


	def timeProcess(self,trnMats):
		mi = 1e16
		ma = 0
		for i in range(len(trnMats)):
			minn = np.min(trnMats[i].data)
			maxx = np.max(trnMats[i].data)
			mi = min(mi, minn)
			ma = max(ma, maxx)
		maxTime = 0
		for i in range(len(trnMats)):
			newData = ((trnMats[i].data - mi) // (3600*24*args.slot)).astype(np.int32)
			maxTime = max(np.max(newData), maxTime)
			trnMats[i] = csr_matrix((newData, trnMats[i].indices, trnMats[i].indptr), shape=trnMats[i].shape)
		print('MAX TIME',mi,ma, maxTime)
		return trnMats, maxTime + 1
	
	def prepareGlobalData(self):
		'''
		adj0 = self.trnMat
		adj0 = (adj0 != 0).astype(np.float32)
		adj0Norm = np.reshape(np.array(np.sum(adj0, axis=1)), [-1])
		for i in range(adj0.shape[0]):
			for j in range(adj0.indptr[i], adj0.indptr[i+1]):
				adj0.data[j] /= adj0Norm[i]
		self.adj = adj0
		'''
		####
		def tran_to_sym(R):
			adj_mat = sp.dok_matrix((args.user + args.item, args.user + args.item), dtype=np.float32)
			adj_mat = adj_mat.tolil()
			R = R.tolil()
			adj_mat[:args.user, args.user:] = R
			adj_mat[args.user:, :args.user] = R.T
			adj_mat = adj_mat.tocsr()
			return (adj_mat+sp.eye(adj_mat.shape[0]))
			

		# adj = self.subMat
		self.maxTime=1
		# self.subMat,self.maxTime=self.timeProcess(self.subMat)
		print(self.subMat[0],self.subMat[-1])
		self.item_with_pop=[]
		for i in range(len(self.sequence)):
			self.item_with_pop.extend(self.sequence[i])
		# print(self.item_with_pop)
		'''
		tpadj = list()
		adjNorm = list()
		tpadjNorm = list()
		for kk in range(args.graphNum):
			adj[kk] = tran_to_sym((adj[kk] != 0).astype(np.float32))
			# tpadj.append(tran_to_sym(transpose(adj[kk])))
			adjNorm.append(np.reshape(np.array(np.sum(adj[kk], axis=1)), [-1]))
			# tpadjNorm[kk] = np.reshape(np.array(np.sum(tpadj[kk], axis=1)), [-1])
			for i in range(adj[kk].shape[0]):
				for j in range(adj[kk].indptr[i], adj[kk].indptr[i+1]):
					adj[kk].data[j] /= adjNorm[kk][i]
		'''
		# self.subadj = adj
		# self.tpsubAdj = tpadj
		# print("adj",adj)
		# self.labelP[i] = np.squeeze(np.array(np.sum(adj, axis=0)))
		# print("labelP",self.labelP)

	def sampleLargeGraph(self, pckUsrs, pckItms=None, sampDepth=2, sampNum=args.graphSampleN, preSamp=False):
		adj = self.adj
		tpadj = self.tpadj
		def makeMask(nodes, size):
			mask = np.ones(size)
			if not nodes is None:
				mask[nodes] = 0.0
			return mask

		def updateBdgt(adj, nodes):
			if nodes is None:
				return 0
			tembat = 1000
			ret = 0
			for i in range(int(np.ceil(len(nodes) / tembat))):
				st = tembat * i
				ed = min((i+1) * tembat, len(nodes))
				temNodes = nodes[st: ed]
				ret += np.sum(adj[temNodes], axis=0)
			return ret

		def sample(budget, mask, sampNum):
			score = (mask * np.reshape(np.array(budget), [-1])) ** 2
			norm = np.sum(score)
			if norm == 0:
				return np.random.choice(len(score), 1), sampNum - 1
			score = list(score / norm)
			arrScore = np.array(score)
			posNum = np.sum(np.array(score)!=0)
			if posNum < sampNum:
				pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
				# pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
				# pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
				pckNodes = pckNodes1
			else:
				pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
			return pckNodes, max(sampNum - posNum, 0)

		def constructData(usrs, itms):
			adj = self.trnMat
			pckU = adj[usrs]
			tpPckI = transpose(pckU)[itms]
			pckTpAdj = tpPckI
			pckAdj = transpose(tpPckI)
			return pckAdj, pckTpAdj, usrs, itms

		usrMask = makeMask(pckUsrs, adj.shape[0])
		itmMask = makeMask(pckItms, adj.shape[1])
		itmBdgt = updateBdgt(adj, pckUsrs)
		if pckItms is None:
			pckItms, _ = sample(itmBdgt, itmMask, len(pckUsrs))
			itmMask = itmMask * makeMask(pckItms, adj.shape[1])
		usrBdgt = updateBdgt(tpadj, pckItms)
		uSampRes = 0
		iSampRes = 0
		for i in range(sampDepth + 1):
			uSamp = uSampRes + (sampNum if i < sampDepth else 0)
			iSamp = iSampRes + (sampNum if i < sampDepth else 0)
			newUsrs, uSampRes = sample(usrBdgt, usrMask, uSamp)
			usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
			newItms, iSampRes = sample(itmBdgt, itmMask, iSamp)
			itmMask = itmMask * makeMask(newItms, adj.shape[1])
			if i == sampDepth or i == sampDepth and uSampRes == 0 and iSampRes == 0:
				break
			usrBdgt += updateBdgt(tpadj, newItms)
			itmBdgt += updateBdgt(adj, newUsrs)
		usrs = np.reshape(np.argwhere(usrMask==0), [-1])
		itms = np.reshape(np.argwhere(itmMask==0), [-1])
		return constructData(usrs, itms)
