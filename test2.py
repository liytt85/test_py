import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 180
PAINT_POINTS = np.vstack([np.linspace(-8,8,ART_COMPONENTS)for _ in range(BATCH_SIZE)])                                                #shape = (64,15)

#print(PAINT_POINTS)

'''plt.plot(PAINT_POINTS[0],2*np.cos(PAINT_POINTS[0])+1,c = '#74BCFF',lw = 3,label='upper bound')
plt.plot(PAINT_POINTS[0],1*np.cos(PAINT_POINTS[0])+0,c = '#FF9359',lw = 3,label='lower bound')
plt.legend(loc = 'upper right')
plt.show()
re = 5*np.power(PAINT_POINTS[0],2)+1 #x^2 *5 +1
#print (re)'''


def artist_works():                                            #即真实的数据
    a = np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]   #shape = (64,1)
    
    paintings = a*np.cos(PAINT_POINTS)+(a-1)
    #print ("panin",paintings)               #shape = (64,15))
    return paintings

with tf.variable_scope('Generator'):                           #使用生成器伪造假的数据
    G_in = tf.placeholder(tf.float32,[None,N_IDEAS])           #shape = (64,5)
    G_l1 = tf.layers.dense(G_in,128,tf.nn.relu)
    G_out = tf.layers.dense(G_l1,ART_COMPONENTS)


with tf.variable_scope('Discriminator'):
    real_art = tf.placeholder(tf.float32,[None,ART_COMPONENTS],name='real_in')  #使用鉴别器来鉴别真实数据
    D_l0 = tf.layers.dense(real_art,128,tf.nn.relu,name='1')                    #并将它判别为1
    prob_artist0 = tf.layers.dense(D_l0,1,name='out')

    #fake art
    D_l1 = tf.layers.dense(G_out,128,tf.nn.relu,name='1',reuse=True)            #使用费鉴别器来判别伪造数据
    prob_artist1 = tf.layers.dense(D_l1,1,name='out',reuse=True)  #并将其判别为0
CLIP = [-0.01, 0.01]
D_loss = tf.reduce_mean(tf.scalar_mul(-1, prob_artist0)+tf.reduce_mean(prob_artist1))          #定义误差函数
G_loss = tf.reduce_mean(tf.scalar_mul(-1, prob_artist1))

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator')
train_D = tf.train.RMSPropOptimizer(LR_D).minimize(                                #定义优化函数
       D_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator'))

train_G = tf.train.RMSPropOptimizer(LR_G).minimize(
       G_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator'))

clip_d_op = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in  d_vars]
with tf.variable_scope('Discriminator'):
	d_loss = tf.summary.scalar("D_loss", D_loss)
for var in d_vars:
	tf.summary.histogram( "ee", var)
with tf.variable_scope('Generator'):
	#tf.summary.histogram("train_G", train_G)
	g_loss = tf.summary.scalar("G_loss", G_loss)
merged = tf.summary.merge_all()
sess= tf.Session()                                                              #初始化流图
sess.run(tf.global_variables_initializer())         
writer1 = tf.summary.FileWriter('log', sess.graph)
plt.ion()
step_div = 50
for step in range(10000):
    artist_paintings = artist_works()
    G_ideas = np.random.randn(BATCH_SIZE,N_IDEAS)
    G_paintings,pa0,new_merge= sess.run([G_out,prob_artist0,merged],
                                  {G_in:G_ideas,real_art:artist_paintings})[:3]
    D1 = sess.run(D_loss, {G_in:G_ideas,real_art:artist_paintings})
    print (D1)
    writer1.add_summary(new_merge, step+1)
    #writer1.add_summary(G1, step+1)
    for i in range(20):
    	train_dd = sess.run(train_D, {G_in:G_ideas,real_art:artist_paintings})
    	#print (train_dd)
    	writer1.add_summary(train_dd, step+1)
    train_g = sess.run(train_G, {G_in:G_ideas})
    writer1.add_summary(train_g, step+1)
    #print (step)
    
    if step%50==0:                                                               #可视化
        plt.cla()
        
        plt.plot(PAINT_POINTS[0],G_paintings[0],c='#4AD631',lw=2,label='Generated painting')
        plt.plot(PAINT_POINTS[0],artist_paintings[0],color='k',lw=2,label='Sample painting')
        plt.plot(PAINT_POINTS[0],2*np.cos(PAINT_POINTS[0])+1,c='#74BCFF',lw=2,label='upper bound')
        plt.plot(PAINT_POINTS[0],1*np.cos(PAINT_POINTS[0])+0,c='#FF9359',lw=2,label='lower bound')
        plt.text(2.5,2.3,'D accuracy=%.2f '%pa0.mean(),fontdict={'size':10})
        plt.text(2.5,2,'D score=%.2f '%-D1,fontdict={'size':10})
        plt.text(2.5,1.7,'step is %.2f'%step,fontdict={'size':10})
        plt.ylim((-3, 5));plt.legend(loc='upper right',fontsize=12);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()

'''import tensorflow as tf 
import numpy as np
import array
class my_first(object):
	def __init__(self, a, b, *args, **kwargs):
		self.first = a
		self.second = b
		print (self.first)
		self.__init(*args, **kwargs)
	def __init(self, k, i, l):
		self.e = k
		self.r = i
		self.o = l
		
		print(self.o)
	def test(self):
		
		print(self.o)
		print("self.o", self.second)
u = my_first(2,3,k=4,i=5,l=6)
u.test()'''

