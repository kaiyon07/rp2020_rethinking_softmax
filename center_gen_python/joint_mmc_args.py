import numpy as np
from scipy.spatial import distance
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--var",type=int,required=True,help="Constant (C)")
parser.add_argument("--dim_dense",type=int,required=True,help=" dimension of the final embedding/dense layer of the model")
parser.add_argument("--num_class",type=int,required=True,help="number of classes in the dataset")
parser.add_argument("--sim",type=int,required=True,help="similarity between the classes (0-1)")
parser.add_argument("--class1",type=int,required=True,help="index of the first class label that needs to be manipulated")
parser.add_argument("--class2",type=int,required=True,help="index of the second class label that needs to be manipulated")
parser.add_argument("--alpha1",type=int,required=True,help="percentage of weight from class1")
parser.add_argument("--alpha2",type=int,required=True,help="percentage of weight from class2")

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

def manipulate(mmc_center,l,u,alpha1,alpha2,similarity):

    score = 0.4*mmc_center[l-1][l-1] + 0.6*mmc_center[u-1][u-1]
    print(score)
    #print(np.linalg.norm(mmc_center[l-1])**2)
    mmc_center[l-1][l-1]= score - similarity
    #print(np.linalg.norm(mmc_center[l-1])**2)
    mmc_center[l-1][u-1]= np.sqrt(1 - np.linalg.norm(mmc_center[l-1])**2)

    mmc_center[u-1][u-1]= score - similarity
    mmc_center[u-1][l-1]=0
    #print(np.linalg.norm(mmc_center[u-1])**2)
    mmc_center[u-1][l-1]=np.sqrt(1 - np.linalg.norm(mmc_center[u-1])**2)

    return mmc_center


var, dim_dense, num_class = args.var, args.dim_dense, args.num_class
mmc_center = generate_mmc_center(var, dim_dense, num_class)

mmc_center_local=mmc_center.copy()

class1, class2, alpha1, alpha2, similarity = args.class1, args.class2, args.alpha1, args.alpha2, args.sim 
final_centers= manipulate(mmc_center_local,class1,class2,alpha1,alpha2,similarity)


before_dist_matrix=distance.cdist(mmc_center,mmc_center , 'euclidean')
after_dist_matrix=distance.cdist(final_centers, final_centers, 'euclidean')

print("Intial distance:", before_dist_matrix)
print("Final distance:", after_dist_matrix)



#import scipy.io as sio
#sio.savemat('./new_params/meanvar'+str(C)+'_featuredim'+str(n_dense)+'_class'+str(class_num)+'.mat', {'mean_logits': mmc_center})