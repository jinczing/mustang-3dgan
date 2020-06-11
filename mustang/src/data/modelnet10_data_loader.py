from torchvision import datasets
from data.data_loader import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

class ModelNet10DataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(datasets.MNIST, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 32768
    
    def save_images(self, images, shape, filename):
        data = images.data if isinstance(images, Variable) else images

        for d in data:
            xs = []
            ys = []
            zs = []
            for i in range(len(d[0])):
                for j in range(len(d[0][i])):
                    for k in range(len(d[0][i][j])):
                        if(int(d[0][i][j][k]) == 1):
                            xs.append(k)
                            ys.append(j)
                            zs.append(i)
            ax = plt.axes(projection='3d')
            ax.scatter(xs=zs, ys=ys, zs=xs)

            savefig(filename, bbox_inches='tight')

class ModelNet10DataSet(Dataset):

    def __init__(self):
    	data = np.load('../modelnet10.npz')
    	X, Y = shuffle(data['X_train'], data['Y_train'])
    	self.X = torch.tensor(X)
    	X = X.reshape(47892, 1, 32, 32, 32)
    	self.Y = torch.tensor(Y)

    def __getitem__(self, index):
    	return self.X[index], self.Y[index]

    def __len__(self):
    	return self.X.shape[0]