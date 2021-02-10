import numpy as np
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--var",type=int,required=True,help="Constant (C_large) optimal around 10")
parser.add_argument("--dim_dense",type=int,required=True,help=" dimension of the final embedding/dense layer of the model")
parser.add_argument("--num_class",type=int,required=True,help="number of classes in the dataset")
parser.add_argument("--var_2",type=int,required=True,help="Constant (C_small) optimal around 1")
parser.add_argument("--num_sup_class",type=int,required=True,help="number of superclasses in the dataset, 20 in case of CIFAR100")


args = parser.parse_args()

def generate_mmc_center(var, dim_dense, num_class): 

    mmc_centers = np.zeros((num_class, dim_dense))
    mmc_centers[0][0] = 1
    for i in range(1,num_class):
        for j in range(i): 
            mmc_centers[i][j] = - (1/(num_class-1) + np.dot(mmc_centers[i],mmc_centers[j])) / mmc_centers[j][j]
        mmc_centers[i][i] = np.sqrt(np.absolute(1 - np.linalg.norm(mmc_centers[i]))**2)
    for k in range(num_class):
        mmc_centers[k] = var * mmc_centers[k]
        
    return mmc_centers


#Ideally for MNIST and CIFAR10: var=10, dim_dense=256, num_class= 10  
var, dim_dense, num_class = args.var, args.dim_dense, args.num_sup_class
mmc_centers_global = generate_mmc_center(var, dim_dense, num_class)

print("mmc_centers shape:",mmc_centers_global.shape)
print("mmc_centers:",mmc_centers_global)


mmc_centers=np.zeros(shape=(args.num_class,args.dim_dense))

var_2, dim_dense, num_class_sub = args.var_2, args.dim_dense, int(args.num_class/args.num_sup_class)
mmc_centers_local = generate_mmc_center(var_2, dim_dense, num_class_sub)
print("opt_means shape:", mmc_centers_local.shape)
print("mmc_centers:",mmc_centers_local)

for i in range(args.num_class):
    mmc_centers[i]=mmc_centers_global[i//num_class_sub]+mmc_centers_local[i%num_class_sub]


print("Final centers shape:", mmc_centers.shape)
print("Final centers:",mmc_centers)

sio.savemat('./hmmc_meanvar'+str(var)+'_'+str(var_2)+'_featuredim'+str(dim_dense)+'_class'+str(num_class)+'.mat', {'mean_logits': mmc_centers})