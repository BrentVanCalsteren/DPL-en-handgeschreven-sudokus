from sudokuNeuralNet.NormalNeuralNets import torchNet
from matplotlib import pyplot as plt
import numpy as np




def main():
    #process_train_data for train data
    #process_querry_data for query data
    time1 = list()
    time2 = list()
    #append_list(filename=f"query_9x9_50sudokus_0images",result=time1)

    """plot2(title="Query time Sudoku 9x9 with constants - dpl aprox",
          legend=["no trained net", "with trained net"],
          list2=[time1,time2],
          list1=[[0,1,20,40],[0,1,20,40]],
          name2="time (sec)", name1="images",
          scale='linear')
    """
    data = list()
    data += torchNet.opendata("train4x4_aprox_16images_test2")
    data += torchNet.opendata("train4x4_aprox_16images_test2_part2")
    iterations_list, loss_list, sudoku_ac_list, image_acc_list, time_list = process_train_data(data)
    data = list()
    data += torchNet.opendata("train4x4_50sudokus_16images_gm_with_help")
    iterations_list2, loss_list2, sudoku_ac_list2, image_acc_list2, time_list2 = process_train_data(data)
    data = list()
    data += torchNet.opendata('train4x4_3val_4images')
    iterations_list3, loss_list3, sudoku_ac_list3, image_acc_list3, time_list3 = process_train_data(data)
    data = list()
    data += torchNet.opendata('train4x4_3val_8images')
    iterations_list4, loss_list4, sudoku_ac_list4, image_acc_list4, time_list4 = process_train_data(data)
    """plot_balk(title="circle permutation accuracy",
          legend=["1->1", "1->2","1->3","1->4"],
          list2=np.array([image_acc_list[-1],image_acc_list2[-1]]).T.tolist())
    """
    """
    plot_balk(title="final image acc ",
          legend=["gm not pretrained", "gm pretrained"],
          list2=[[image_acc_list[-1][0],image_acc_list2[-1][0]],[image_acc_list3[-1],image_acc_list4[-1]]])
    """
    #"""
    plot2(title="iteration time 4x4 16 images",
      legend=["gm","gm pretrained"],
      list2=[[18,28,58,155,235,291]],
      list1=[[2,4,8,12,24,28]],
      name2="time (sec)", name1="iteration",
      scale='linear')
    """#"""
    """
    #data = [torchNet.opendata('offset_acc8')]
    data.append(torchNet.opendata('offset_acc8_part2'))
    plot_balk(title="circle permutation accuracy (8 images 4x4 sudoku)",
          legend=["1->1", "1->2","1->3","1->4"],
          list2=np.array(data).T.tolist())
    """#"""


def append_list(filename="",data=list(),result=list()):
    data.append(torchNet.opendata(filename))
    result.append(average(process_querry_data(data[-1])))
    return data, result


def average(l):
    return sum(l) / len(l)


def plot_balk(title="", list2=None, list1=None, legend=None):
    if not list1:
        list1 = ["250 iter","500 iter"]

    plt.bar([1,2],list2[0],width=0.2,color="b")
    plt.bar([1.2,2.2], list2[1],width=0.2,color="lightblue")
    plt.bar([1.4,2.4], list2[2], width=0.2,color="skyblue")
    plt.bar([1.6, 2.6], list2[3], width=0.2,color="steelblue")
    plt.title(title)
    plt.ylabel("accuracy")
    if legend:
        plt.legend(legend)
    plt.show()

def plot2(title="", list2=None, list1=None, name2="", name1="iterations", legend=None, scale='linear'):
    if not list1:
        list1 = list()
        for i in range(len(list2)):
            list1.append(create_list_from_bound(50,len(list2[i]),scale=50))

    a = list()
    colour = ["darkblue","darkorange","darkgreen"]
    for i in range(len(list2)):
        """b = [list2[i][0][0],list2[i][1][0],list2[i][2][0],list2[i][3][0],list2[i][4][0]]
        plt.scatter(list1[i], b, marker="_")
        a.append(np.polyfit(list1[i], b, deg=1))"""
        #plt.scatter(list1[i], list2[i], marker="_")
        a.append(np.polyfit(list1[i], list2[i], deg=1))
        plt.plot(list1[i], list2[i])
    """
    for i,el in enumerate(a):
        xs = np.linspace(0, 500, num=100)
        plt.plot(xs, np.polyval(el, xs),'--', color=colour[i],linewidth = '1')
    """#"""
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.yscale(scale)
    #plt.xlim(0, 100)
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


def create_list_from_bound(r, l,scale=1):
    if (r == l):
        return [r]
    else:
        res = []
        while (r < l*scale + 1):
            res.append(r)
            r += scale
        return res


if __name__ == '__main__':
    main()

