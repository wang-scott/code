import mainprogram
from PIL import Image
import numpy as np
import csv
import random
from skimage import io
from invert_LSB_module import *
import os

def authorize(name, len_r=4, len_b=4, alpha=2):
    # 初始化陣列
    U_second = []
    U_second_un = []
    detect_image = np.zeros((512, 512, 3))
    # 指定 CSV 檔案名稱
    csv_filename = f'list/{name}.csv'
    
    # 從 CSV 檔案讀取資料
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 跳過表頭
        next(reader)
        # 將資料放入陣列
        for row in reader:
            if row[0].strip() != "":
                U_second.append(int(row[0]))
            if row[1].strip() != "":
                U_second_un.append(int(row[1]))
    authentication_code = []
    ac = []
    with open('authentication_code.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            ac.append(row)
            
    print("U_second:", U_second)
    print("U_second_un:", U_second_un)
 
    stego_img = io.imread(f'embeding_noise/{name}.png')
    I = Image.open(f'processed_image/{name}.png').convert('RGB')
    I = np.array(I)
    origin_image = np.array(Image.open(f'image/{name}.tiff').convert('RGB'))
    authentication_code = hash_all_pixel(origin_image,len_r,len_b) 
    gray_img = rgb2gray(stego_img)
    bin_matrix = dec2bin(stego_img)
    # ac = np.array(ac)
    # if ac == authentication_code:
    #     print('identical')
    # else:
    #     print('not identical')
    len_bb = (len_r + len_b) // 2

    # 開始驗證
    detected_error = 0
    divid4 = (stego_img//(2**len_r))*(2**len_r)
    count1 = count2 = count3 = 0
    for i in range(stego_img.shape[0]):
        for j in range(stego_img.shape[1]):
            ii = i * stego_img.shape[1] + j + 1
            detect_image[i][j] = (0, 0, 0)
            flag = True
            if gray_img[i][j] in U_second_un:
                count1 += 1
                ac1 = bin_matrix[i][j][2][6:] #取出的驗證碼
                ac2 = dec2bin(gggg(stego_img[i][j], ii, alpha))[2][6:] #算出的驗證碼
                if ac1 == ac2:
                    detect_image[i][j] = (255, 255, 255) 
                    flag = False   
            elif gray_img[i][j] in U_second:
                count2 += 1
                ac1 = bin_matrix[i][j][2][8-len_bb:] 
                ac2 = dec2bin(proposed_unsolvable_case(stego_img[i][j], ii, len_bb))[2][8-len_bb:]
                if ac1 == ac2:
                    detect_image[i][j] = (255, 255, 255)
                    flag = False
            else:
                count3 += 1
                a = Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],length=len_r)
                origin_a = int(Embedding(dec2bin(origin_image[i][j])[2],authentication_code[i][j][:len_r],length=len_r),2)
                b = Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],1,length=len_r)
                origin_b = int(Embedding(dec2bin(origin_image[i][j])[2],authentication_code[i][j][:len_r],1,length=len_r),2)
                ac_b = eee(origin_a,origin_b,stego_img[i][j][2])
                c = Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],length=len_b)
                origin_c = int(Embedding(dec2bin(origin_image[i][j])[0],authentication_code[i][j][len_r:],length=len_b),2)
                d = Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],1,length=len_b)
                origin_d = int(Embedding(dec2bin(origin_image[i][j])[0],authentication_code[i][j][len_r:],1,length=len_b),2)
                ac_r = eee(origin_c,origin_d,stego_img[i][j][0])

                current_r = bin_matrix[i][j][0][len_r:]
                current_b = bin_matrix[i][j][2][len_b:]
                if (c[len_r:] == current_r or d[len_r:] == current_r)  and (a[len_b:] == current_b or b[len_b:] == current_b):
                    detect_image[i][j] = (255, 255, 255)
                    flag = False
                # if not flag:
                #     if (I[i,j] != stego_img[i,j]).any():
                #         print(f"像素驗證錯誤：位置 ({i}, {j})")
                #         raise RuntimeError(f"像素驗證錯誤：位置 ({i}, {j})")
            #assert flag == False,"error"
            if flag:
                detected_error += 1

    print(f"Detected error: {detected_error}")
    print(f"Count1: {count1}, Count2: {count2}, Count3: {count3}")
    diff_pixels = 0
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if(I[i,j] != np.array(stego_img[i,j])).any():
               diff_pixels+=1 

    accuracy = detected_error/diff_pixels
    print(f"Detected error: {detected_error}, Actual error: {diff_pixels}, Accuracy: {accuracy}")
    return detected_error
    
def embeding(image,n):
    def noise(I,Noise):
        n_r,n_c = Noise.shape[0],Noise.shape[1]

        r_base = random.randint(0,I.shape[0]-n_r)
        c_base = random.randint(0,I.shape[1]-n_c)
        for i in range(n_r):
            for j in range(n_c):
                if(Noise[i, j,3]==0):
                    continue
                for k in range(3):
                    I[i+r_base,j+c_base,k] = Noise[i,j,k]

        return I
    I=image.copy()
    path2 = "noise/"+n+".png"
    I2=io.imread(path2)
    e = noise(I,I2)
    io.imshow(e)
    #io.show() 
    return e
    
if __name__ == "__main__":
    # 開啟影像並轉換為 NumPy 陣列
    
    image = np.array(Image.open('image/jet.tiff'))
    name = 'bean'
    Stego, _ = mainprogram.propose_main(name,image, 4, 4, 2)
    Stego = np.array(Image.open(f'processed_image/{name}.png').convert('RGB'))
    error_image = embeding(Stego, 'rock')
    io.imsave(f'embeding_noise/{name}.png', error_image.astype(np.uint8))
    authorize(name)