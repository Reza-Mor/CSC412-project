import matplotlib.pyplot as plt
import numpy as np

file = open("result.txt", "r")
result  = file.read()

def drawLearningCurve(recode):
    names = recode.keys()
    for name in names:
        plt.plot(recode[name], '--',label = "task: "+str(name))
    plt.ylabel('Accuracy')

    plt.legend()
    plt.xlabel("Interation")
    plt.show()

def drawHistagram(task1_result, task2_result, task3_result):
    bar_width = 0.2
    index = np.array(range(len(task1_result)))
    rects1 = plt.bar(index, task1_result, bar_width,  label='task0')
    rects2 = plt.bar(index+bar_width, task2_result, bar_width,  label='task1')
    rects3 = plt.bar(index+2*bar_width, task3_result, bar_width,  label='task2')
    plt.xticks(index + bar_width, ['finetune', 'ewc', 'offline'])  #add new agent here
    plt.ylim(ymax=100, ymin=0)
    plt.ylabel('Performance')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    test = {0:[2.994943857192993, 2.9940185546875,  2.9814703464508057,
               2.8723857402801514, 2.657487392425537, 2.502932548522949,
               2.139519453048706, 2.4010727405548096, 2.1511080265045166],
            1:[2.6893320083618164,2.382406234741211,2.30564284324646,
               2.0461392402648926,1.96821928024292,2.030503511428833,
               1.9104994535446167,1.8491305112838745,1.9131875038146973],
            2:[1.9826408624649048, 1.8170222043991089, 1.7612231969833374,
               1.652228832244873, 1.7717312574386597, 1.690726637840271,
               1.513128638267517, 1.7446374893188477, 1.6601581573486328]}
    drawLearningCurve(test)


    task1_result = [0.8, 0.5, 0.6]   #with respect to the oder of agents above
    task2_result = [0.4, 0.25, 0.3]
    task3_result = [0.2, 0.2, 0.25]
    drawHistagram(task1_result, task2_result, task3_result)