import tensorflow as tf
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--var",type=int,required=True,help="Constant (C)")
parser.add_argument("--dim_dense",type=int,required=True,help=" dimension of the final embedding/dense layer of the model")
parser.add_argument("--num_class",type=int,required=True,help="number of classes in the dataset")
parser.add_argument("--lr",type=int,required=True,help="learning rate")
parser.add_argument("--steps",type=int,required=True,help="number of optimization steps usually around 100000")
args = parser.parse_args()

L = args.num_class         # 10 Number of classes
d = args.dim_dense         # 2 Dimension of features
lr = args.lr               # 0.0001 Learning rate
mean_var = args.var        # 10
steps = args.steps         # 100000 optimization steps

z = tf.get_variable("auxiliary_variable", [d, L]) #dxL
x = z / tf.norm(z, axis=0, keepdims=True) #dxL, normalized in each column
XTX = tf.matmul(x, x, transpose_a=True) - 2 * tf.eye(L)#LxL, each element is the dot-product of two means, the diag elements are -1
cost = tf.reduce_max(XTX) #single element
opt = tf.train.AdamOptimizer(learning_rate=lr)
opt_op = opt.minimize(cost)
with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    for i in range(steps):
        _, loss = sess.run([opt_op, cost])
        min_distance2 = loss
        print('Step %d, min_distance2: %f'%(i, min_distance2))


    mean_logits = sess.run(x)

mean_logits = mean_var * mean_logits.T 
print(mean_logits)

sio.savemat('./isometric_meanvar1_featuredim'+str(d)+'_class'+str(L)+'.mat', {'mean_logits': mean_logits})