import numpy as np
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--var",type=int,required=True,help="Constant (C)")
parser.add_argument("--dim_dense",type=int,required=True,help=" dimension of the final embedding/dense layer of the model")
parser.add_argument("--num_class",type=int,required=True,help="number of classes in the dataset")
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

#Ideally for MNIST and CIFAR10: var=10, dim_dense=256, num_classes= 10  
var, dim_dense, num_class = args.var, args.dim_dense, args.num_class
mmc_centers = generate_mmc_center(var, dim_dense, num_class)

print(mmc_centers.shape)
print(mmc_centers)

sio.savemat('./meanvar'+str(var)+'_featuredim'+str(dim_dense)+'_class'+str(num_class)+'.mat', {'mean_logits': mmc_centers})