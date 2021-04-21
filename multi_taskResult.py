import matplotlib.pyplot as plt
import numpy as np

def draw(task1_result, task2_result, task3_result):
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


if __name__ == '__main__':
    task1_result = [80, 58, 60]   #with respect to order of agents above
    task2_result = [40, 25, 30]
    task3_result = [20, 20, 25]
    draw(task1_result, task2_result, task3_result)