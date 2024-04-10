import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from load_data import *


def tSNE2dim(data):
    tsne = TSNE(n_components = 2)
    result = copy.deepcopy(tsne.fit_transform(data))
    return result


def plotdata2dim(data, label, title="t-SNE Visualization", figname="t_sne_clustering.png"):
    data2dim = tSNE2dim(data)
    plt.scatter(data2dim[:, 0], data2dim[:, 1], c=label)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    plt.colorbar()
    # plt.show()
    plt.savefig(figname)
    plt.clf()
    print('========== {} is save in fig =========='.format(title))


def plotrawfeatures():
    dataset = ['cora', 'pubmed']
    for setname in dataset:
        data, adj, label = load_data(setname)
        plotdata2dim(data, label, title="t-SNE Visualization of "+setname+" (Raw features)", figname="fig/tsne_raw_"+setname)


def plotloss(loss, title="Loss", figname="loss.png"):
    iterations = range(1, len(loss) + 1)
    plt.plot(iterations, loss, )
    plt.xlabel("Iteration")
    plt.ylabel("Loss value")
    plt.title(title)
    # plt.show()
    plt.savefig(figname)
    plt.clf()
    print('========== {} is save in fig =========='.format(title))


def plotACCandNMI(alpha, ACC, NMI, title="Impact of alpha", figname="Impact of alpha"):
    fig, ax1 = plt.subplots()

    ax1.plot(alpha, ACC, 'b-o')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('ACC/%', color='b')

    ax2 = ax1.twinx()
    ax2.plot(alpha, NMI, 'r-d')
    ax2.set_ylabel('NMI/%', color='r')

    plt.title(title)
    # plt.show()
    plt.savefig(figname)
    plt.clf()
    print('========== {} is save in fig =========='.format(title))

# # Test ACCandNMI
# alpha = np.linspace(0, 1, 20)
# ACC = np.random.rand(20, 1)
# NMI = np.random.rand(20, 1)
# plotACCandNMI(alpha, ACC, NMI)

