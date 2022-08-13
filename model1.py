from ast import arg
from Params import args
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import Utils.TimeLogger as logger
import numpy as np
from Utils.TimeLogger import log
from DataHandler import negSamp,negSamp_fre, transpose, DataHandler, transToLsts, posSamp
from Utils.attention import AdditiveAttention,MultiHeadSelfAttention
import scipy.sparse as sp
class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
		self.positions_indexs=list(range(100))

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()
	# def LightGcn(self, adj, )
	'''
	def messagePropagate(self, lats, adj):
		# GAT=GraphAttentionLayer(lats.shape[-1],lats.shape[-1],adj,args.user+args.item)
		# return GAT.call(lats)
		return Activate(tf.sparse.sparse_dense_matmul(adj, lats), self.actFunc)
		# return tf.sparse.sparse_dense_matmul(adj, lats)
	def hyperPropagate(self, lats, adj):
		# return adj @ (tf.transpose(adj) @ lats)
		# return FC(adj @ Activate(tf.transpose(adj) @ lats, self.actFunc), args.latdim, activation=self.actFunc)
		return Activate(adj @ Activate(tf.transpose(adj) @ lats, self.actFunc), self.actFunc)
	# act(user * item * item * dim) = user * dim
	# FC(dim*item)=dim*hyperNum
	# hyperNum*dim + user*dim
	# 这里多一层有用，dropout没用
	# return user_dim
	def hyperPropagate(self, lats, adj):
		lat1 = Activate(tf.transpose(adj) @ lats, self.actFunc)
		lat2 = tf.transpose(FC(tf.transpose(lat1), args.hyperNum, activation=self.actFunc)) + lat1
		lat3 = tf.transpose(FC(tf.transpose(lat2), args.hyperNum, activation=self.actFunc)) + lat2
		lat4 = tf.transpose(FC(tf.transpose(lat3), args.hyperNum, activation=self.actFunc)) + lat3
		# lat5 = tf.transpose(FC(tf.transpose(lat4), args.hyperNum, activation=self.actFunc)) + lat3
		ret = Activate(adj @ lat4, self.actFunc)
		# print("LATS",lat1,lat2,lat3,lat4,ret)
		# ret = adj @ lat4
		return ret
	'''
	def edgeDropout(self, mat):
		def dropOneMat(mat):
			# print("drop",mat)
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			# newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
			newVals = tf.nn.dropout(values, self.keepRate)
			return tf.sparse.SparseTensor(indices, newVals, shape)
		return dropOneMat(mat)
	# cross-view collabrative Supervision
	def ours(self):
		# 用来存每一个short term的graph的学习结果
		user_vector,item_vector=list(),list()
		# embedding
		uEmbed=NNs.defineParam('uEmbed', [args.graphNum, args.user, args.latdim], reg=True)
		iEmbed=NNs.defineParam('iEmbed', [args.graphNum, args.item, args.latdim], reg=True)	
		posEmbed=NNs.defineParam('posEmbed', [args.pos_length, args.latdim], reg=True)
		# graphNum是short term的数量，每个short graph做一次lightgcn的学习
		for k in range(args.graphNum):
			embs0=[uEmbed[k]]
			embs1=[iEmbed[k]]
			for i in range(args.gnn_layer):
				# 因为graph的形式是[user+item,user+item]，所以把user的embedding和item的embedding concat一下
				embs=tf.concat([embs0[-1],embs1[-1]],axis=0)
				# subAdj[k]就是第k个short term graph
				all_emb = Activate(tf.sparse.sparse_dense_matmul(self.edgeDropout(self.subAdj[k]), embs), self.actFunc)
				# 把user和item的特征矩阵分开
				a_emb0,a_emb1=tf.split(all_emb, [args.user, args.item], axis=0)
				# 上一跳学的特征+这一跳学的
				embs0.append(a_emb0+embs0[-1]) 
				embs1.append(a_emb1+embs1[-1]) 
			# 对每一跳学完的特征求和
			user=tf.add_n(embs0)
			item=tf.add_n(embs1)
			# 然后再加一个全连接层，就是第k个short term的graph的结果
			user_vector.append(FC(user,outDim=args.latdim))
			item_vector.append(FC(item,outDim=args.latdim))
		# now user_vector is [g,u,latdim]
		user_vector=tf.stack(user_vector,axis=0)
		item_vector=tf.stack(item_vector,axis=0)

		# user_vector is [u,g,latdim]
		user_vector_tensor=tf.transpose(user_vector, perm=[1, 0, 2])
		item_vector_tensor=tf.transpose(item_vector, perm=[1, 0, 2])
		'''
		# 对所有short term的graph学出来的item和user的特征矩阵求和再加一个全连接层，得到long term的表征
		final_user_vector = FC(tf.reduce_sum(user_vector,axis=0),outDim=args.latdim)
		final_item_vector = FC(tf.reduce_sum(item_vector,axis=0),outDim=args.latdim)
		'''		
		self.additive_attention0 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.additive_attention1 = AdditiveAttention(args.query_vector_dim,args.latdim)
		# self.multihead_self_attention0 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		# self.multihead_self_attention1 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		# multihead_user_vector = self.multihead_self_attention0.attention(user_vector_tensor)
		# multihead_item_vector = self.multihead_self_attention1.attention(item_vector_tensor)
		# final_user_vector = self.additive_attention0.attention(multihead_user_vector/8+user_vector_tensor)		
		# final_item_vector = self.additive_attention1.attention(multihead_item_vector/8+item_vector_tensor)
		final_user_vector = self.additive_attention0.attention(user_vector_tensor)		
		final_item_vector = self.additive_attention1.attention(item_vector_tensor)
		# final_user_vector = FC(tf.reduce_sum(multihead_user_vector,axis=1),outDim=args.latdim)
		# final_item_vector = FC(tf.reduce_sum(multihead_item_vector,axis=1),outDim=args.latdim)

		'''
		# 本来用attention的 但是效果调不好。。。。
		self.additive_attention0 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.additive_attention1 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.multihead_self_attention0 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		self.multihead_self_attention1 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		
		final_user_vector=list()
		final_item_vector=list()
		i=0
		while(i<args.user):
			multihead_user_vector = self.multihead_self_attention0.attention(user_vector_tensor[i:i+args.batch_size])
			final_user_vector0 = self.additive_attention0.attention(multihead_user_vector)# (user_vector_tensor)
			final_user_vector.extend(final_user_vector0)
			i+=args.batch_size
		final_user_vector=tf.stack(final_user_vector,axis=1)
		i=0
		while(i<args.item):
			multihead_item_vector = self.multihead_self_attention1.attention(item_vector_tensor[i:i+args.batch_size])
			final_item_vector0 = self.additive_attention1.attention(multihead_item_vector)# (item_vector_tensor)
			final_item_vector.extend([final_item_vector0])
			i+=args.batch_size
		final_item_vector=tf.stack(final_item_vector,axis=0)
		'''
		# 基于long term的pred计算
		pckUlat = tf.nn.embedding_lookup(final_user_vector, self.uids)
		pckIlat = tf.nn.embedding_lookup(final_item_vector, self.iids)
		if(self.is_train==True):	
			pckIlat += tf.nn.embedding_lookup(posEmbed, self.posids)
		preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)
		if(self.is_train==True):	
			preds_short=list()
			for k in range(args.graphNum):
				preds_short.append(tf.reduce_sum(tf.nn.embedding_lookup(user_vector[k], self.uids) * tf.nn.embedding_lookup(item_vector[k], self.iids),axis=-1))
			preds_short_value=list()
			for j in range(tf.shape(self.uids)[0].eval()):
				preds_short_value.append(preds_short[self.timeids[j]][j])
			preds_short_value=tf.stack(preds_short_value,axis=0)
			preds+=preds_short_value
		'''
		elif (self.is_train==False):	
			preds_short=list()
			for k in range(args.graphNum):
				preds_short.append(tf.reduce_sum(tf.nn.embedding_lookup(user_vector[k], self.uids) * tf.nn.embedding_lookup(item_vector[k], self.iids),axis=-1))
			preds_short_value=list()
			for j in range(tf.shape(self.uids)[0].eval()):
				preds_short_value.append(preds_short[-1][j])
			preds_short_value=tf.stack(preds_short_value,axis=0)
			preds+=preds_short_value
		'''
		# 开始计算对比学习的损失
		sslloss = 0	
		# print(S_final)
		def get_cos_distance(X1, X2, sampNam):
			# calculate cos distance between two sets
			# more similar more big
			# 求模
			X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
			X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
			# 内积
			mask = tf.diag(tf.ones([2*sampNam]))
			X1_X2 = tf.matmul(tf.matmul(X1, tf.transpose(X2)),mask)
			X1_X2_norm = tf.matmul(tf.reshape(X1_norm,[-1,1]),tf.reshape(X2_norm,[1,-1]))
			# 计算余弦距离
			cos = X1_X2/X1_X2_norm
			return cos
		self.preds_one=list()
		self.final_one=list()
		# print(S_final)
		# suids是对每一个短期的图随机抽取的边，用来对比学习
		for i in range(args.graphNum):
			sampNum = tf.shape(self.suids[i])[0] // 2 # pair的数量
			pckUlat = tf.nn.embedding_lookup(final_user_vector, self.suids[i])
			pckIlat = tf.nn.embedding_lookup(final_item_vector, self.siids[i])
			# 计算来自long term的S^
			S_final = tf.reduce_sum(Activate(pckUlat* pckIlat + pckUlat + pckIlat, self.actFunc),axis=-1)
			# S_final += tf.reduce_sum(Activate(get_cos_distance(pckUlat,pckIlat,sampNum), self.actFunc),axis=-1)
			posPred_final = tf.slice(S_final, [0], [sampNum])
			negPred_final = tf.slice(S_final, [sampNum], [-1])
			S_final = posPred_final-negPred_final
			pckUlat = tf.nn.embedding_lookup(user_vector[i], self.suids[i])
			pckIlat = tf.nn.embedding_lookup(item_vector[i], self.siids[i])
			# 计算来自short term的S
			preds_one = tf.reduce_sum(Activate(pckUlat* pckIlat + pckUlat + pckIlat, self.actFunc), axis=-1)
			# preds_one += tf.reduce_sum(Activate(get_cos_distance(pckUlat,pckIlat,sampNum), self.actFunc),axis=-1)
			posPred = tf.slice(preds_one, [0], [sampNum])
			negPred = tf.slice(preds_one, [sampNum], [-1])
			# 计算(S1^-S2^)(S1-S2)
			sslloss += tf.reduce_sum(tf.maximum(0.0, 1.0 -S_final * (posPred-negPred)))
			self.preds_one.append(preds_one)
			self.final_one.append(sslloss)
		return preds, sslloss

	def prepareModel(self):
		self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
		self.is_train = tf.placeholder(dtype=tf.float32, shape=[])
		NNs.leaky = args.leaky
		self.actFunc = 'leakyRelu'
		'''
		adj = self.handler.trnMat
		idx, data, shape = transToLsts(adj, norm=True)
		self.adj = tf.sparse.SparseTensor(idx, data, shape)
		idx, data, shape = transToLsts(transpose(adj), norm=True)
		print("idx,data,shape",idx,data,shape)
		self.tpAdj = tf.sparse.SparseTensor(idx, data, shape)
		'''
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
		self.posids = tf.placeholder(name='posids', dtype=tf.int32, shape=[None])
		self.timeids = tf.placeholder(name='timeids', dtype=tf.int32, shape=[None])
		self.suids=list()
		self.siids=list()
		for k in range(args.graphNum):
			self.suids.append(tf.placeholder(name='suids%d'%k, dtype=tf.int32, shape=[None]))
			self.siids.append(tf.placeholder(name='siids%d'%k, dtype=tf.int32, shape=[None]))
		self.subAdj=list()
		self.subAdjNp=list()
		for i in range(args.graphNum):
			seqadj = self.handler.subadj[i]
			idx, data, shape = transToLsts(seqadj, norm=True)
			self.subAdj.append(tf.sparse.SparseTensor(idx, data, shape))
			self.subAdjNp.append(sp.lil_matrix(self.handler.subadj[i]).toarray())
		'''
		增加一个训练时获得所属短期图的信息，加起来
		先试一下循环加 看看会不会很慢
		seqadj = self.handler.trnMat.astype(np.int32)# timeMat
		idx, data, shape = transToLsts(seqadj, norm=True)
		self.timeMat=tf.sparse.SparseTensor(idx, data, shape)
		'''		
		#############################################################################
		self.preds, self.sslloss = self.ours()
		sampNum = tf.shape(self.uids)[0] // 2
		posPred = tf.slice(self.preds, [0], [sampNum])# begin at 0, size = sampleNum
		negPred = tf.slice(self.preds, [sampNum], [-1])# 
		# self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
		self.regLoss = args.reg * Regularize()  + args.ssl_reg * self.sslloss
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
		trnPos = self.handler.trnPos[batIds]
		# print(labelMat)
		# temLabel=labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * train_sample_num
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		positionLocs = [None] * temlen
		timeLocs = [None] * temlen
		cur = 0				
		for i in range(batch):
			# posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(train_sample_num, len(labelMat[batIds[i]]))
			if sampNum == 0:
				poslocs = [np.random.choice(args.item)]
				neglocs = [poslocs[0]]
			else:
				poslocs=posSamp(labelMat[batIds[i]],sampNum-1).tolist()[0]
				# poslocs = np.random.choice(posset, sampNum)
				# poslocs = list(np.random.choice(posset, sampNum-1))
				poslocs.extend([trnPos[i]])
				neglocs = negSamp(labelMat[batIds[i]], sampNum, args.item, trnPos[i])
			# print(poslocs)
			for j in range(sampNum):
				# print(j)
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				timeLocs[cur] = timeMat[batIds[i],posloc]
				timeLocs[cur+temlen//2] = timeMat[batIds[i],negloc]
				cur += 1
			positionLocs[cur-sampNum:cur]=self.positions_indexs[:cur]
		'''
		# 每一对正负例对应的user是一样的
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		'''
		timeLocs = timeLocs[:cur] + timeLocs[temlen//2: temlen//2 + cur]
		positionLocs = positionLocs[:cur] + [args.pos_length-1]*cur
		# print(uLocs[0],uLocs[1],iLocs[0],iLocs[1])
		return uLocs, iLocs, timeLocs, positionLocs

	def sampleSslBatch(self, batIds, labelMat):
		temLabel=list()
		for k in range(args.graphNum):	
			# print(labelMat[k][batIds])
			# print(labelMat[k][batIds][args.user:])
			temLabel.append(labelMat[k][batIds].toarray())
		batch = len(batIds)
		temlen = batch * 2 * args.sslNum
		uLocs = [[None] * temlen] * args.graphNum
		iLocs = [[None] * temlen] * args.graphNum
		for k in range(args.graphNum):	
			cur = 0				
			for i in range(batch):
				posset = np.reshape(np.argwhere(temLabel[k][i]!=0), [-1])
				# print(posset)
				sslNum = min(args.sslNum, len(posset)//2)
				if sslNum == 0:
					poslocs = [np.random.choice(args.item)]
					neglocs = [poslocs[0]]
				else:
					all = np.random.choice(posset, sslNum*2) - args.user
					# print(all)
					poslocs = all[:sslNum]
					neglocs = all[sslNum:]
				for j in range(sslNum):
					posloc = poslocs[j]
					negloc = neglocs[j]
					'''					
					# 相同用户的内部对比
					uLocs[k][cur] = uLocs[k][cur+temlen//2] = batIds[i]
					iLocs[k][cur] = posloc
					iLocs[k][cur+temlen//2] = negloc
					cur += 1
					'''
					# 随机对比
					uLocs[k][cur] = uLocs[k][cur+1] = batIds[i]
					iLocs[k][cur] = posloc
					iLocs[k][cur+1] = negloc
					cur += 2
			'''
			# 每一对正负例对应的user是一样的
			uLocs[k] = uLocs[k][:cur] + uLocs[k][temlen//2: temlen//2 + cur]
			iLocs[k] = iLocs[k][:cur] + iLocs[k][temlen//2: temlen//2 + cur]
			'''
			uLocs[k]=uLocs[k][:cur]
			iLocs[k]=iLocs[k][:cur]
			

		# print(uLocs[0],uLocs[1],iLocs[0],iLocs[1])
		return uLocs, iLocs

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		sample_num_list=[40]		
		steps = int(np.ceil(num / args.batch))
		for s in range(len(sample_num_list)):
			for i in range(steps):
				st = i * args.batch
				ed = min((i+1) * args.batch, num)
				batIds = sfIds[st: ed]

				target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.preds, self.final_one, self.preds_one]
				feed_dict = {}
				uLocs, iLocs, timeLocs, posLocs = self.sampleTrainBatch(batIds, self.handler.user_sequence, self.handler.timeMat, sample_num_list[s])
				suLocs, siLocs = self.sampleSslBatch(batIds, self.handler.subadj)
				feed_dict[self.uids] = uLocs
				feed_dict[self.iids] = iLocs
				feed_dict[self.timeids] = timeLocs
				feed_dict[self.is_train] = True
				for k in range(args.graphNum):
					feed_dict[self.suids[k]] = suLocs[k]
					feed_dict[self.siids[k]] = siLocs[k]
				# print(len(suLocs),len(siLocs))
				feed_dict[self.keepRate] = args.keepRate

				res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

				preLoss, regLoss, loss, pre, finalS, pone = res[1:]
				if(np.isnan(regLoss)):
					kk=0
					while(kk<len(finalS)):
						# print(pre[kk:kk+10])
						print('finalS',finalS[k])
						print('pred one',pone[k])
						kk+=1

				epochLoss += loss
				epochPreLoss += preLoss
				log('Step %d/%d: preloss = %.2f, REGLoss = %.2f         ' % (i+s*steps, steps*len(sample_num_list), preLoss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batIds, labelMat): # labelMat=TrainMat(adj)
		batch = len(batIds)
		temTst = self.handler.tstInt[batIds]
		temLabel = labelMat[batIds].toarray()
		# print("temLabel",temLabel)
		temlen = batch * 100
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			rdnNegSet = negSamp_fre(temLabel[i], 99, self.handler.neg_sequency)
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uLocs[cur] = batIds[i]
				iLocs[cur] = locset[j]
				cur += 1
		return uLocs, iLocs, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		epochHit5, epochNdcg5 = [0] * 2
		epochHit20, epochNdcg20 = [0] * 2
		epochHit1, epochNdcg1 = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))
		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			feed_dict = {}
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMat)
			suLocs, siLocs = self.sampleSslBatch(batIds, self.handler.subadj)
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.is_train] = False
			for k in range(args.graphNum):
				feed_dict[self.suids[k]] = suLocs[k]
				feed_dict[self.siids[k]] = siLocs[k]
			feed_dict[self.keepRate] = 1.0
			preds = self.sess.run(self.preds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			if(i==20):
				kk=0
				print(preds)
				# while(kk<len(preds)):
				# 	print(preds[kk:kk+10])
				# 	kk+=10
			hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1 = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			epochHit5 += hit5
			epochNdcg5 += ndcg5
			epochHit20 += hit20
			epochNdcg20 += ndcg20
			epochHit1 += hit1
			epochNdcg1 += ndcg1
			log('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		print("epochNdcg1,epochHit1,epochNdcg5,epochHit5,epochNdcg20,epochHit20",epochNdcg1/ num,epochHit1/ num,epochNdcg5/ num,epochHit5/ num,epochNdcg20/ num,epochHit20/ num)
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		hit1 = 0
		ndcg1 = 0
		hit5=0
		ndcg5=0
		hit20=0
		ndcg20=0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
			shoot = list(map(lambda x: x[1], predvals[:1]))
			if temTst[j] in shoot:
				hit1 += 1
				ndcg1 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))	
			shoot = list(map(lambda x: x[1], predvals[:5]))
			if temTst[j] in shoot:
				hit5 += 1
				ndcg5 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
			shoot = list(map(lambda x: x[1], predvals[:20]))	
			if temTst[j] in shoot:
				hit20 += 1
				ndcg20 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))	
		return hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '4.his', 'wb') as fs:# 上次是5
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '4.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	