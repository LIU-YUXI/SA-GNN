from scipy.sparse import *
 
row =  [0,0,0,1,1,1,2,2,2]#行指标
col =  [0,1,2,0,1,2,0,1,2]#列指标
data = [1,0,1,0,1,1,1,1,0]#在行指标列指标下的数字
team = csr_matrix((data,(row,col)),shape=(3,3))
print(team[2,2])


