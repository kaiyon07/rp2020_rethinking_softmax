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

var, dim_dense, num_class = 10, 256, 20
mmc_centers_global = generate_mmc_center(var, dim_dense, num_class)
print("mmc_centers global shape- ", mmc_centers_global.shape)
print("mmc_centers global:",mmc_centers_global)


final_centers=np.zeros(shape=(100,256))

var_2, dim_dense_l, sub_class = 1, 256, 5
mmc_centers_local = generate_mmc_center(var_2, dim_dense_l, sub_class)
print("mmc center local shape- ", mmc_centers_local.shape)
print("mmc_centers local:",mmc_centers_local)

for i in range(100):
    final_centers[i]=mmc_centers_global[i//5]+mmc_centers_local[i%5]


print(final_centers)




sio.savemat('./cifar100_hmmc_tree_featuredim'+str(dim_dense)+'_class'+str(num_class)+'.mat', {'mean_logits': final_centers})