from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import matplotlib.pyplot as plt
from multiprocessing import Manager, freeze_support
import numpy as np
from time import perf_counter
import cv2


def worker_s(img, new_img, rescaleFactor):
    org_size = img.shape
    new_size = [i*rescaleFactor for i in org_size]

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            x = int(i/rescaleFactor)
            y = int(j/rescaleFactor)
            new_img[i][j] = img[x][y]

def worker_p(img, new_img, rescaleFactor, id, num_threads):
    org_size = img.shape
    new_size = [i*rescaleFactor for i in org_size]

    for i in range(int(id*(new_size[0]/num_threads)),int((id+1)*(new_size[0]/num_threads))):
        for j in range(new_size[1]):
            x = int(i/rescaleFactor)
            y = int(j/rescaleFactor)
            new_img[i][j] = img[x][y]

    return(new_img)

def nearestNeighbourInterpolation(img, rescaleFactor, num_threads):
    img_size = img.shape
    new_size = [i*rescaleFactor for i in img_size]
    new_img_s = np.zeros(new_size)
    new_img_p = np.zeros(new_size)

    start_s=perf_counter()
    worker_s(img,new_img_s,rescaleFactor)
    end_s=perf_counter()
    time_s = end_s-start_s

    start_p = perf_counter()
    taskList = []

    with ProcessPoolExecutor(max_workers=num_threads) as pool:
        for id in range(num_threads):
            taskList.append(pool.submit(worker_p,img,new_img_p,rescaleFactor,id,num_threads))

        wait(taskList)
        for task in taskList:
            new_img_p += task.result()

    end_p = perf_counter()
    time_p = end_p-start_p

    return(new_img_s,new_img_p,time_s,time_p)

if (__name__=='__main__'):
    num_threads = 4
    img_size = (256,256)
    rescaleFactor = 16
    # img = np.random.randint(0,high=256,size=img_size,dtype=np.uint8)
    img = cv2.imread("./image.jpg", 0)

    new_img_s,new_img_p,time_s,time_p = nearestNeighbourInterpolation(img, rescaleFactor, num_threads)

    print("Serial Time: "+str(round(time_s,3))+"s")
    print("Parallel Time: "+str(round(time_p,3))+"s")


    new_img_s = new_img_s.astype(np.uint8)
    new_img_p = new_img_p.astype(np.uint8)

    print(img)
    print(new_img_p)

    cv2.imshow("org_img",img)
    cv2.imshow("img_s",new_img_s)
    cv2.imshow("img_p",new_img_p)
    cv2.waitKey(0)
    cv2.destroyAllWindows()