
from tslearn.clustering import TimeSeriesKMeans
import pickle
import numpy as np
from multiprocessing import Process

def Clustering(num_cluster):
    print("clustering ",num_cluster)
    kmeans_total_3 = np.ndarray([int(155/5),int(155/5)], dtype='object')
    newSeries_total_3 = np.ndarray([int(155/5),int(155/5)], dtype='object')
    rese_total_3 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')
    kluster_model_3 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')
    centroids_3 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')

    kmeans_total_4 = np.ndarray([int(155/5),int(155/5)], dtype='object')
    newSeries_total_4 = np.ndarray([int(155/5),int(155/5)], dtype='object')
    rese_total_4 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')
    kluster_model_4 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')
    centroids_4 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')

    kmeans_total_6 = np.ndarray([int(155/5),int(155/5)], dtype='object')
    newSeries_total_6 = np.ndarray([int(155/5),int(155/5)], dtype='object')
    rese_total_6 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')
    kluster_model_6 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')
    centroids_6 = np.ndarray([int(155/5),int(155/5),num_cluster], dtype='object')

    for a in range(0,155,5):
        for b,index in zip(range(a+5,155,5),range(0,int((155-a-5)/5))):
            new_series_3 = allPixelNDVIPoly3[:,a:b]
            new_series_4 = allPixelNDVIPoly4[:,a:b]
            new_series_6 = allPixelNDVIPoly6[:,a:b]

            km_3 = TimeSeriesKMeans(n_clusters=num_cluster, metric="euclidean", max_iter=25,random_state=0)
            km_4 = TimeSeriesKMeans(n_clusters=num_cluster, metric="euclidean", max_iter=25,random_state=0)
            km_6 = TimeSeriesKMeans(n_clusters=num_cluster, metric="euclidean", max_iter=25,random_state=0)

            kluster_model_3[int(a/5),index] = km_3
            kluster_model_4[int(a/5),index] = km_4
            kluster_model_6[int(a/5),index] = km_6

            if(new_series_3.shape[1] != 0):
                y_pred_3 = km_3.fit_predict(new_series_3)
                kmeans_total_3[int(a/5),index] = y_pred_3
                newSeries_total_3[int(a/5),index] = new_series_3
                for p in range(0,num_cluster):
                    rese_total_3[int(a/5),index,p] = newResa3[y_pred_3 == p]
                    centroids_3[int(a/5),index,p] = km_3.cluster_centers_[p].ravel()

            if(new_series_4.shape[1] != 0):
                y_pred_4 = km_4.fit_predict(new_series_4)
                kmeans_total_4[int(a/5),index] = y_pred_4
                newSeries_total_4[int(a/5),index] = new_series_4
                for p in range(0,num_cluster):
                    rese_total_4[int(a/5),index,p] = newResa4[y_pred_4 == p]
                    centroids_4[int(a/5),index,p] = km_4.cluster_centers_[p].ravel()
    
            if(new_series_6.shape[1] != 0):
                y_pred_6 = km_6.fit_predict(new_series_6)
                kmeans_total_6[int(a/5),index] = y_pred_6
                newSeries_total_6[int(a/5),index] = new_series_6
                for p in range(0,num_cluster):
                    rese_total_6[int(a/5),index,p] = newResa6[y_pred_6 == p]
                    centroids_6[int(a/5),index,p] = km_6.cluster_centers_[p].ravel()

    with open('./pickles/kmeans_total_3_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(kmeans_total_3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/rese_total_3_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(rese_total_3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/kluster_model_3_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(kluster_model_3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/kluster_centroids_3_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(centroids_3, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./pickles/kmeans_total_4_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(kmeans_total_4, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/rese_total_4_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(rese_total_4, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/kluster_model_4_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(kluster_model_4, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/kluster_centroids_4_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(centroids_4, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./pickles/kmeans_total_6_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(kmeans_total_6, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/rese_total_6_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(rese_total_6, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/kluster_model_6_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(kluster_model_6, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./pickles/kluster_centroids_6_c'+str(num_cluster)+'.pickle', 'wb') as handle:
        pickle.dump(centroids_6, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("clustering ",num_cluster," completed!")

#Caricamento dei dati e rese
with open('./pickles/allPixelNDVIPoly.pickle', 'rb') as handle:
    allPixelNDVIPoly3 = pickle.load(handle)

with open('./pickles/allPixelNDVIPoly4.pickle', 'rb') as handle:
    allPixelNDVIPoly4 = pickle.load(handle)

with open('./pickles/allPixelNDVIPoly6.pickle', 'rb') as handle:
    allPixelNDVIPoly6 = pickle.load(handle)

with open('./pickles/newResa3.pickle', 'rb') as handle:
    newResa3 = pickle.load(handle)

with open('./pickles/newResa4.pickle', 'rb') as handle:
    newResa4 = pickle.load(handle)

with open('./pickles/newResa6.pickle', 'rb') as handle:
    newResa6 = pickle.load(handle)

allPixelNDVIPoly3 = allPixelNDVIPoly3[(newResa3<=11000) & (newResa3 >= 4000),:]
newResa3 = newResa3[(newResa3<=11000) & (newResa3 >= 4000)]

allPixelNDVIPoly4 = allPixelNDVIPoly4[(newResa4<=11000) & (newResa4 >= 4000),:]
newResa4 = newResa4[(newResa4<=11000) & (newResa4 >= 4000)]
print(len(newResa4))

allPixelNDVIPoly6 = allPixelNDVIPoly6[(newResa6<=11000) & (newResa6 >= 4000),:]
newResa6 = newResa6[(newResa6<=11000) & (newResa6 >= 4000)]
print(len(newResa6))

#Clustering



# Global
#num_cluster = 2
if __name__ == '__main__':
    # k-means per tutti le combinazioni di intervalli temporali 
    processList = []
    for num_cluster in range(2,7):
        processList.append(Process(target=Clustering, args=(num_cluster,)))
        #Clustering(num_cluster)
    for p in processList:
        p.start()
    
    for p in processList:
        p.join()
    print("clustering completed")
