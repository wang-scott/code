import mainprogram
from PIL import Image
import numpy as np
import csv
import random
from skimage import io
from invert_LSB_module import *
import os
import pandas as pd

def cal_PSNR(name,dir_name='processed_image'):
    # 讀取圖片
    img1 = cv2.imread(f'{dir_name}/{name}.png')
    img2 = cv2.imread(f'image/{name}.tiff')

    # 計算 MSE
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        print("兩張圖片完全相同")
    else:
        # 計算 PSNR
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        print(f"PSNR: {psnr} dB")
    return psnr

def ablation(name,img,len_r,len_b,alpha):
    gray_img = rgb2gray(img)
    img_divid4 = img//4*4#去除1,2的LSB(除以4)
    bin_matrix = dec2bin(img)
    authentication_code = hash_all_pixel(img,len_r,len_b)
    second_matrix = np.zeros((img.shape)) 
    proposed_embedded_num = 0 
    len_bb= (len_b+len_r)//2
    for i in range(bin_matrix.shape[0]):
        for j in range(bin_matrix.shape[1]):
            a = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],length=len_r),2)#red
            #b = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],1,length=len_r),2)#red
            second_matrix[i][j][2] = a
            c = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],length=len_b),2)#blue
            #d = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],1,length=len_b),2)#blue
            second_matrix[i][j][0] = c
    U_second = cal_green(second_matrix,gray_img)
    U_second_un=[]
    for target in U_second:
        results = find_all_values(gray_img, target)
        for i,j in results:
            if proposed_unsolvable_case(img[i][j],i*gray_img.shape[1]+j+1,len_bb=len_bb)==None:
                U_second_un.append(gray_img[i][j])
    U_second_un = list(set(U_second_un))
    
    max_len = max(len(U_second), len(U_second_un))
    U_second += [""] * (max_len - len(U_second))
    U_second_un += [""] * (max_len - len(U_second_un))
    
    #輸出U_sendond和U_second_un到csv
    os.makedirs('list', exist_ok=True) 
    csv_filename = f"list/{name}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["usecond", "usecondun"])
        for u, uu in zip(U_second, U_second_un):
            writer.writerow([u, uu])
        
    # print(f'{U_second=}')
    # print(f'{U_second_un=}')
    #embedding第一種方法使用lsb顛倒 第二種方法使用len_bb=(len_r+len_b)//2 and 2
    tmp=0           
    tmp1=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ii = i*gray_img.shape[1]+j+1
            if gray_img[i][j] in U_second_un:#嵌入長度只有2
                second_matrix[i][j]=gggg(img[i][j],ii,alpha)#propose方法
                proposed_embedded_num += 2
                tmp1+=1
            elif gray_img[i][j] in U_second: #使用第二種方法
                second_matrix[i][j]=proposed_unsolvable_case(img[i][j],ii,len_bb)#propose方法
                proposed_embedded_num += (len_r+len_b)/2
                tmp+=1
            else : #使用第一種方法
                second_matrix[i][j][1]=round((gray_img[i][j]-0.299*second_matrix[i][j][2]-0.114*second_matrix[i][j][0])/0.587)
                proposed_embedded_num += len_r+len_b
    

    # print('Proposed unsolvable pixel(len_r+len_b)/2:',tmp,'len2=',tmp1)
    dir_name = 'ablation_test_image'
    os.makedirs(dir_name, exist_ok=True) 
    io.imsave(f'{dir_name}/{name}.png', second_matrix.astype(np.uint8))
    psnr = cal_PSNR(name,'ablation_test_image')
    return  second_matrix.astype(np.uint8),proposed_embedded_num

lr = 4
imagelist = ['tree']

for name in imagelist:
    result = []
    for lr in [2, 3, 4]:
        image = np.array(Image.open(f'image/{name}.tiff').convert('RGB'))
        Stego, payload = ablation(name, image, lr, lr, 2)
        psnr = cal_PSNR(name, 'ablation_test_image')
        result.append({
            "name": name,
            "lr": lr,
            "psnr": psnr,
            "payload": payload
        })
    df = pd.DataFrame(result, columns=["name", "lr", "psnr", "payload"])
    df.to_csv(f"ablation_test_data/{name}.csv", index=False, encoding="utf-8-sig")