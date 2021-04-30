import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from pandas import DataFrame
from sklearn.cluster import KMeans





length = 10000

def x1(s):
	g1x = np.random.normal(5, s)
	return g1x
def x2(s):
	g2x = np.random.normal(20, s)
	return g2x 
def y1(s):
	g1y = np.random.normal(2, s)
	return g1y 
def y2(s):
	g2y = np.random.normal(14, s)
	return g2y 
def x3(s):
        g3x = np.random.normal(-4, s)
        return g3x 
def y3(s):
        g3y = np.random.normal(-7, s)
        return g3y 

print("5, 2")
print("20, 14")



xlist = [] #generating random x cooridantes for H0
for i in range(1, length):
	d1 = x1(5)
	xlist.append(d1)
	d2 = x2(3)
	xlist.append(d2)


ylist = [] #generating random x cooridnate for H1
for i in range(1, length): 
	d4 = y1(4)
	ylist.append(d4)
	d5 = y2(1)
	ylist.append(d5)

print(len(ylist),len(xlist))


Data = {'x': xlist,
        'y': ylist
        }



#plt.figure()
#plt.scatter(g1x, g2y, color = 'blue')
#plt.scatter(g2x, g2y, color = 'red')
#plt.show()


df = DataFrame(Data, columns=['x','y'])

#Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
#        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
#       }
  
#df = DataFrame(Data,columns=['x','y'])

kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.figure()
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel("x")
plt.ylabel("y")
#plt.savefig("cluster10000.png")
plt.show()

