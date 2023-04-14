import matplotlib.pyplot as plt


def plot(numbers1=[], numbers2=[], label1='', label2=''):
    plt.plot(numbers1, numbers2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.show()
