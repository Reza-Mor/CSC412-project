import matplotlib.pyplot as plt
import numpy as np

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


def drawTimeCruve(x, y1, y2, label1, label2):
    plt.plot(x,y1, label=label1)
    plt.plot(x,y2, label=label2)
    plt.xlabel("Time")
    plt.legend()
    plt.show()








if __name__ == '__main__':
    task1_result = [80, 58, 60]   #with respect to order of agents above
    task2_result = [40, 25, 30]
    task3_result = [20, 20, 25]
    #drawHistagram(task1_result, task2_result, task3_result)

    test = {1:[2.994943857192993, 2.9940185546875,  2.9814703464508057,
               2.8723857402801514, 2.657487392425537, 2.502932548522949,
               2.139519453048706, 2.4010727405548096, 2.1511080265045166],
            2:[2.6893320083618164,2.382406234741211,2.30564284324646,
               2.0461392402648926,1.96821928024292,2.030503511428833,
               1.9104994535446167,1.8491305112838745,1.9131875038146973],
            3:[1.9826408624649048, 1.8170222043991089, 1.7612231969833374,
               1.652228832244873, 1.7717312574386597, 1.690726637840271,
               1.513128638267517, 1.7446374893188477, 1.6601581573486328]}
    #drawLearningCurve(test)
    #drawTimeCruve([0,1,2,3,4], [0,1,2,3,4],"label")