import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def drawLearningCurve(recode):
    names = recode.keys()
    for name in names:
        plt.plot(recode[name], '--',label = "task: "+str(name))
    plt.ylabel('Loss')

    plt.legend()
    plt.xlabel("Interation")
    plt.show()


def drawHistagram(task1_result, task2_result, task3_result):
    bar_width = 0.2
    index = np.array(range(len(task1_result)))
    rects1 = plt.bar(index, task1_result, bar_width,  label='task1')
    rects2 = plt.bar(index+bar_width, task2_result, bar_width,  label='task2')
    rects3 = plt.bar(index+2*bar_width, task3_result, bar_width,  label='task3')
    plt.xticks(index + bar_width, ['finetune', 'ewc', 'offline'])  #add new agent here
    plt.ylim(ymax=100, ymin=0)
    plt.ylabel('Performance')

    plt.legend()
    plt.show()

def get_res(file):
    with open(file) as f:
        lines = f.readlines()
    res = {}
    for line in lines:
        if line.startswith('Training'):
            task = int(line[15])
        if line.startswith('EPOCH'):
            epoch = int(line[7:])
        if line.startswith('	Task'):
            eval_task = int(line[6])
            if (task, epoch, eval_task) not in res:
                res[(task, epoch, eval_task)] = 0
            res[(task, epoch, eval_task)] += float(line[17:])
    for k in res:
        res[k] = res[k] / 3
    return res

def get_acc(file):
    with open(file) as f:
        lines = f.readlines()
    ls = []
    for line in lines:
        if line.startswith('Training'):
            task = int(line[15])
        if line.startswith('EPOCH'):
            epoch = int(line[7:])
        if line.startswith('	Task'):
            if epoch == 29:
                ls.append(float(line[17:]))
    return [np.mean(ls[:9]), np.mean(ls[9:18]), np.mean(ls[18:])]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


if __name__ == '__main__':
    '''
    plt.bar([str(i+1) for i in range(20)], [0.0036, 0.0056, 0.0092, 0.0114, 0.0139, 0.0164, 0.0204, 0.0426, 0.0770, 0.4413, 0.0828, 0.1668, 0.0431, 0.0278, 0.0146, 0.0088, 0.0018, 0.0025, 0.0103, 0.0000])
    plt.xlabel('Class')
    plt.ylabel('fraction')
    plt.savefig('class_dist.png')
    '''

    acc_finetune = get_acc('gaussian_gr.log')
    print(mean_confidence_interval(acc_finetune))
    acc_ewc = get_acc('gaussian_gr.log')
    print(mean_confidence_interval(acc_ewc))

    '''
    res_finetune = get_res('filter_finetune.log')
    res_ewc = get_res('filter_ewc.log')
    res_gr = get_res('filter_gr.log')

    fig, axes = plt.subplots(3, 3, figsize=(6, 5))

    ls = []
    for i in range(3):
        for j in range(3):
            l_fintune = []
            l_ewc = []
            l_gr = []
            for k in range(2, 30, 3):
                l_fintune.append(res_finetune[(i, k, j)])
                l_ewc.append(res_ewc[(i, k, j)])
                l_gr.append(res_gr[(i, k, j)])
            ax_1 = axes[j][i].plot(range(2, 30, 3), l_fintune, color='red')
            ax_2 = axes[j][i].plot(range(2, 30, 3), l_ewc, color='blue')
            ax_3 = axes[j][i].plot(range(2, 30, 3), l_gr, color='green')
            if j < 2:
                axes[j][i].get_xaxis().set_visible(False)
            else:
                axes[j][i].set_xlabel('Train {} (Epoch)'.format(i+1))
            if i > 0:
                axes[j][i].get_yaxis().set_visible(False)
            else:
                axes[j][i].set_ylabel('Task {} (Acc)'.format(j+1))
            axes[j][i].set_ylim([0, 80])
            axes[j][i].title.set_visible(False)
    fig.legend([ax_1, ax_2, ax_3], labels=['Finetune', 'EWC', 'GR'])
    plt.subplots_adjust(wspace=0.03, hspace=0.1)
    plt.savefig('t.png')
    '''

