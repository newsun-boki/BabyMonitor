import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

root_dir = "./output"
file_name = "/detect_datas.csv"
circles = np.loadtxt(root_dir + "/2023-03-07_23-13-47_circle" + file_name,delimiter=',')
yupdowns = np.loadtxt(root_dir + "/2023-03-08_01-17-35_UpDown" + file_name,delimiter=',')
rightlefts = np.loadtxt(root_dir + "/2023-03-23_22-11-18_RightLeft" + file_name,delimiter=',')
zupdowns = np.loadtxt(root_dir + "/2023-03-23_22-47-14_ZUpDown" + file_name,delimiter=',')
data = np.concatenate([circles,yupdowns,rightlefts,zupdowns],axis=0)
print(data.shape)

# tsne = TSNE(n_components=3, random_state=42)
# data_tsne = tsne.fit_transform(data)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

circles_size = circles.shape[0]
yupdowns_size = yupdowns.shape[0]
rightlefts_size = rightlefts.shape[0]
zupdowns_size = zupdowns.shape[0]

# ax.scatter(data_tsne[:circles_size, 0], data_tsne[:circles_size, 1], data_tsne[:circles_size, 2], label='Circles')
# ax.scatter(data_tsne[circles_size:circles_size+yupdowns_size, 0], data_tsne[circles_size:circles_size+yupdowns_size, 1], data_tsne[circles_size:circles_size+yupdowns_size, 2], label='UpDowns')
# ax.scatter(data_tsne[circles_size+yupdowns_size:circles_size+yupdowns_size+rightlefts_size, 0], data_tsne[circles_size+yupdowns_size:circles_size+yupdowns_size+rightlefts_size, 1], data_tsne[circles_size+yupdowns_size:circles_size+yupdowns_size+rightlefts_size, 2], label='RightLefts')
# ax.scatter(data_tsne[circles_size+yupdowns_size+rightlefts_size:, 0], data_tsne[circles_size+yupdowns_size+rightlefts_size:, 1], data_tsne[circles_size+yupdowns_size+rightlefts_size:, 2], label='ZUpDowns')

# ax.set_title("t-SNE Visualization")
# ax.set_xlabel("Dimension 1")
# ax.set_ylabel("Dimension 2")
# ax.set_zlabel("Dimension 3")
# ax.legend()

# plt.show()
import umap.umap_ as umap

reducer = umap.UMAP(n_components=3, random_state=42)
data_umap = reducer.fit_transform(data)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_umap[:circles_size, 0], data_umap[:circles_size, 1], data_umap[:circles_size, 2], label='Circles')
ax.scatter(data_umap[circles_size:circles_size+yupdowns_size, 0], data_umap[circles_size:circles_size+yupdowns_size, 1], data_umap[circles_size:circles_size+yupdowns_size, 2], label='UpDowns')
ax.scatter(data_umap[circles_size+yupdowns_size:circles_size+yupdowns_size+rightlefts_size, 0], data_umap[circles_size+yupdowns_size:circles_size+yupdowns_size+rightlefts_size, 1], data_umap[circles_size+yupdowns_size:circles_size+yupdowns_size+rightlefts_size, 2], label='RightLefts')
ax.scatter(data_umap[circles_size+yupdowns_size+rightlefts_size:, 0], data_umap[circles_size+yupdowns_size+rightlefts_size:, 1], data_umap[circles_size+yupdowns_size+rightlefts_size:, 2], label='ZUpDowns')

ax.set_title("UMAP Visualization")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
ax.legend()

plt.show()
