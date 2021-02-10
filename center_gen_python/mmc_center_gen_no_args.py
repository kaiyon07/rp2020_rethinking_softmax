import numpy as np
import scipy.io as sio

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

var, dim_dense, num_class = 10 , 256 , 10
mmc_centers = generate_mmc_center(var, dim_dense, num_class)

print("MMC centers shape:", mmc_centers.shape)
print("MMC centers:", mmc_centers)



sio.savemat('./meanvar'+str(var)+'_featuredim'+str(dim_dense)+'_class'+str(num_class)+'.mat', {'mean_logits': mmc_centers})