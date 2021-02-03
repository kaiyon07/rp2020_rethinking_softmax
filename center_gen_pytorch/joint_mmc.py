import numpy as np
from scipy.spatial import distance

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


var, dim_dense, num_class = 1, 256, 10
mmc_center = generate_mmc_center(var, dim_dense, num_class)

mmc_center_local=mmc_center.copy()

final_centers= manipulate(mmc_center_local,4,6,0.4,0.6,0.1)


before_dist_matrix=distance.cdist(mmc_center,mmc_center , 'euclidean')
after_dist_matrix=distance.cdist(final_centers, final_centers, 'euclidean')

print("Intial distance:", before_dist_matrix[3,5])
print("Final distance:", after_dist_matrix[3,5])



#import scipy.io as sio
#sio.savemat('./new_params/meanvar'+str(C)+'_featuredim'+str(n_dense)+'_class'+str(class_num)+'.mat', {'mean_logits': mmc_center})