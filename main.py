# 输入：按时间划分的t个adj图，每个图都有几乎所有的用户和商品，那么每个图应该是一样的大小：item*user
# t个gnn模块,输出ui1..uit特征矩阵,ii1...iit特征矩阵
# self att得ui ii
# preloss
# 算了先处理数据吧
# 先八张图
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
from model import Recommender
if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()