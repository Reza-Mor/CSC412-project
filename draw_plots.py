import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2

def drawLearningCurve(recode):
    names = recode.keys()
    for name in names:
        plt.plot(recode[name], '--',label = "task: "+str(name))
    plt.ylabel('Loss')

    plt.legend()
    plt.xlabel("Interation")
    plt.show()


def get_acc(line):
    return float(line[line.index(':')+1:-1]) 

def readResult():
    dict={}
    key=''
    num=-1
    result = open("result.txt", 'r')
    for line in result:
        line=line.strip()
        if line and not 'Task' in line:
            num=-1
            key = line
            dict.update({key:[]})
        elif 'Task 0' in line:
            num+=1
            dict[key].append([get_acc(line)])
        elif line:
            dict[key][num].append(get_acc(line))
    print(dict)
    return dict 


def arrange_dict():
    dict=readResult()
    gaussian={0:[[],[],[]],1:[[],[],[]],  2:[[],[],[]]}
    filter={0:[[],[],[]],1:[[],[],[]],  2:[[],[],[]]}
    for key in dict.keys():
        if 'gaussian' in key:
            for i in range(len(dict[key])):
                gaussian[i][0].append(dict[key][i][0])
                gaussian[i][1].append(dict[key][i][1])
                gaussian[i][2].append(dict[key][i][2])
   
        elif 'filter' in key:
            for l in range(len(dict[key])):
                filter[l][0].append(dict[key][l][0])
                filter[l][1].append(dict[key][l][1])
                filter[l][2].append(dict[key][l][2])
    return  gaussian, filter


def drawAllHistagram():
    gaussian, filter = arrange_dict()
    for i in range(len(gaussian)):
        drawHistagram(gaussian[i][0],gaussian[i][1], gaussian[i][2], "Gaussian"+str(i))

    for i in range(len(filter)):
        drawHistagram(filter[i][0],filter[i][1], filter[i][2], "Filter"+str(i))

    dispayimages()
    





def drawHistagram(task1_result, task2_result,task3_result, filename):
    bar_width = 0.2
    index = np.array(range(len(task1_result)))
    rects1 = plt.bar(index, task1_result, bar_width,  label='task1')
    rects2 = plt.bar(index+bar_width, task2_result, bar_width,  label='task2')
    rects3 = plt.bar(index+2*bar_width, task3_result, bar_width,  label='task3')
    plt.xticks(index + bar_width, ['finetune', 'ewc', 'generative replay'])  #add new agent here
    plt.ylim(ymax=100, ymin=0)
    plt.ylabel('Performance')
    plt.title(filename)
    plt.legend()
    plt.savefig("histgrams/"+filename)
    plt.show()
   


def drawTimeCruve(x, y1, y2, label1, label2):
    plt.plot(x,y1, label=label1)
    plt.plot(x,y2, label=label2)
    plt.xlabel("Time")
    plt.legend()
    plt.show()


def dispayimages():
     gauss0 = cv2.imread('histgrams/Gaussian0.png')
     gauss1 = cv2.imread("histgrams/Gaussian1.png")
     gauss2 = cv2.imread("histgrams/Gaussian2.png")
     gauss_concate_Hozi=cv2.hconcat([gauss0, gauss1,gauss2])
     filter0 = cv2.imread("histgrams/Filter0.png")
     filter1 = cv2.imread("histgrams/Filter1.png")
     filter2 = cv2.imread("histgrams/Filter2.png")
     filter_concate_Hozi=cv2.hconcat([filter0,filter1,filter2])
     concate_together = cv2.vconcat([gauss_concate_Hozi,filter_concate_Hozi])
     cv2.imwrite('histgrams/all_histagrams.png', concate_together)



    








if __name__ == '__main__':
    drawAllHistagram()
