from matplotlib.pyplot import legend
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def tsne_graph_output(w,h,tdm,vocab):
    tsne = TSNE(random_state=3211)
    tsne_embedding = tsne.fit_transform(w)
    tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x', 'y'])
    tsne_embedding['hue'] = w.argmax(axis=1)

    legend_list = []
    topics = ["topic "+str(i) for i in range(h.shape[0])]

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.1, wspace=.001)
    data = tsne_embedding
    scatter = plt.scatter(data=data,x='x',y='y',s=6,c=data['hue'],cmap="Set1")
    plt.axis('off')
    plt.show()
    print("")
    colors = []
    for i in range(len(topics)):
       idx = np.where(data['hue']==i)[0][0]
       color = scatter.get_facecolors()[idx]
       colors.append(color)
       legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))


    print()