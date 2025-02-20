import matplotlib.pyplot as plt

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

def plot_train_val(train_losses, val_losses, filename):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.savefig(filename)

if __name__ == '__main__':
    plot_train_val([1,1,2,3,1], [2,3,4,2,5], "out.jpg")
