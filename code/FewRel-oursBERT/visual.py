from sklearn.manifold import TSNE
from sklearn import manifold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

temp = np.load('temp.npy')
labels = np.load('labels.npy')
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(temp)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure()
for i in range(X_norm.shape[0]):
    #plt.scatter()
    plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]), 
             fontdict={'weight': 'bold', 'size': 9})
#plt.xticks([])
#plt.yticks([])
#plt.show()
plt.savefig('ours-bert.jpg')