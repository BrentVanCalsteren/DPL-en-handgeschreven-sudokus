from sudokuNeuralNet.NormalNeuralNets import torchNet
from matplotlib import pyplot as plt




def main():
    #process_train_data for train data
    #process_querry_data for query data

    data = list()
    time_list = list()
    #iterations_list,loss_list,sudoku_ac_list,image_acc_list,time_list = process_train_data(data)

    for i in range(47):
        append_list(f'querry9x9_time{i}image_1solutions',data=data,result=time_list)

    plot2(title="query time 9x9 dpl exact 1 solution - linear scale",
          #legend=["no images"],
          list2=[time_list], name2="time (sec)", name1="images in sudoku",
          scale='linear')


def append_list(filename="",data=list(),result=list()):
    data.append(torchNet.opendata(filename))
    result.append(average(process_querry_data(data[-1])))


def average(l):
    return sum(l) / len(l)


def plot2(title="", list2=None, list1=None, name2="", name1="iterations", legend=None, scale='linear'):
    if not list1:
        list1 = list()
        for i in range(len(list2)):
            list1.append(create_list_from_bound(0,len(list2[i])-1))

    for i in range(len(list2)):
        plt.plot(list1[i], list2[i])

    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.yscale(scale)
    plt.title(title)
    if legend:
        plt.legend(legend)
    plt.show()



def process_querry_data(data):
    time_list = list()
    l = subtract_prev(data)
    time_list += l
    return time_list


def process_train_data(data):
    iterations_list = list()
    loss_list = list()
    sudoku_ac_list = list()
    image_acc_list = list()
    time_list = list()
    last_iteration = 0

    for el in data:
        last_iteration, l = add_iter(list(el[0].keys()),last_iteration)
        iterations_list += l
        l = subtract_prev(list(el[0].values()))
        time_list += l
        loss_list += list(el[1].values())
        sudoku_ac_list.append(el[2])
        image_acc_list.append(el[3])
    return iterations_list,loss_list,sudoku_ac_list,image_acc_list,time_list


def add_iter(l,last):
    x = 0
    a = list()
    for el in l:
        el = int(el)
        el+=(last-1)
        x = el
        a.append(el)
    return x,a


def subtract_prev(l):
    prev = 0
    x = 0
    a = list()
    for el in l:
        x = el
        el-=prev
        prev = x
        a.append(el)
    return a


def create_list_from_bound(r, l):
    if (r == l):
        return [r]
    else:
        res = []
        while (r < l + 1):
            res.append(r)
            r += 1
        return res


if __name__ == '__main__':
    main()

