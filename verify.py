import mainprogram
from PIL import Image
import numpy as np
import csv
import random
from skimage import io
from invert_LSB_module import *
import os
import cv2

def generateU(img,len_r,len_b):
    bin_matrix = dec2bin(img)
    gray_img = rgb2gray(img)
    authentication_code = hash_all_pixel(img,len_r,len_b)   

    second_matrix = np.zeros((img.shape)) 
    for i in range(bin_matrix.shape[0]):
        for j in range(bin_matrix.shape[1]):
            a = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],length=len_r),2)#red
            b = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],1,length=len_r),2)#red
            second_matrix[i][j][2] = eee(a,b,img[i][j][2])
            c = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],length=len_b),2)#blue
            d = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],1,length=len_b),2)#blue
            second_matrix[i][j][0] = eee(c,d,img[i][j][0])
            
    U_second = cal_green(second_matrix,gray_img)
    U_second_un=[]
    for target in U_second:
        results = find_all_values(gray_img, target)
        for i,j in results:
            if proposed_unsolvable_case(img[i][j],i*gray_img.shape[1]+j+1,len_bb=4)==None:
                U_second_un.append(gray_img[i][j])
    U_second_un = list(set(U_second_un))
    return U_second,U_second_un

def authorize(origin,img,len_r,len_b):
    U_second,U_second_un = generateU(origin,len_r,len_b)
    # U_second=[]
    # U_second_un=[]
 
    bin_matrix = dec2bin(img)
    gray_img = rgb2gray(img)

    new_auth_code = hash_all_pixel(img,len_r,len_b) 
    len_bb = (len_b+len_r)//2

    embedded_tamper_matrix = np.zeros((img.shape)).astype(np.uint8) 
    aaa,bbb,ccc=0,0,0
    for i in range(img.shape[0]):
        if i%50 == 0:
            print(i)
        for j in range(img.shape[1]):
            ii = i*gray_img.shape[1]+j+1
            if gray_img[i][j] in U_second_un:
                # tt = generate_perturbed_pairs(img[i][j],ii,alpha)!= img[i][j]
                # if True in tt :
                #     embedded_tamper_matrix[i][j]=255
                b,g,r = img[i][j] 
                tt = hash_unsolvable(ii,gray_img[i][j],r,g,b//4*4,2)
                if b != int(Embedding(str(dec2bin(b)),tt,length=2),2):
                    embedded_tamper_matrix[i][j]=255
                    aaa+=1
            elif gray_img[i][j] in U_second :
                # tt = proposed_unsolvable_case(img[i][j],ii,len_bb) != img[i][j]
                # if True in tt  :
                #     embedded_tamper_matrix[i][j]=255
                b,g,r = img[i][j]
                tt = proposed_hash_unsolvable(ii,gray_img[i][j],(b//(2**len_bb))*(2**len_bb),len_bb)
                if b != int(Embedding(str(dec2bin(b)),tt,length=len_bb),2):
                    embedded_tamper_matrix[i][j]=255
                    bbb+=1
                    # print(b,int(Embedding(str(dec2bin(b)),tt,length=len_bb),2))
            else :
                a = int(Embedding(bin_matrix[i][j][2],new_auth_code[i][j][:len_r],length=len_r),2)#red
                b = int(Embedding(bin_matrix[i][j][2],new_auth_code[i][j][:len_r],1,length=len_r),2)#red
                c = int(Embedding(bin_matrix[i][j][0],new_auth_code[i][j][len_r:],length=len_b),2)#blue
                d = int(Embedding(bin_matrix[i][j][0],new_auth_code[i][j][len_r:],1,length=len_b),2)#blue
                rr = eee(a,b,img[i][j][2])
                bb = eee(c,d,img[i][j][0])             
                if  img[i][j][2]!=rr or img[i][j][0]!=bb :
                    embedded_tamper_matrix[i][j]=255
                    ccc+=1
                    # print(origin[i][j],img[i][j],new_auth_code[i][j],origin_auth_code[i][j])
    print(aaa,bbb,ccc)
    return embedded_tamper_matrix, aaa+bbb+ccc   

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
    return e
    
if __name__ == "__main__":
    # 開啟影像並轉換為 NumPy 陣列
    name = 'Peppers'
    image = np.array(Image.open(f'image/{name}.tiff').convert('RGB'))
    
    Stego, _ = mainprogram.propose_main(name,image, 4, 4, 2)
    Stego = np.array(Image.open(f'processed_image/{name}.png'))
    error_image = embeding(Stego, 'tomato')
    io.imsave(f'embeding_noise/{name}.png', error_image)
    origin = np.array(Image.open(f'image/{name}.tiff').convert('RGB'))
    error_image = np.array(Image.open(f'embeding_noise/{name}.png').convert('RGB'))
    detect_image, error = authorize(origin,error_image,4,4)

    actual = 0
    processed_image = np.array(Image.open(f'processed_image/{name}.png').convert('RGB'))
    for i in range(origin.shape[0]):
        for j in range(origin.shape[1]):
            if (processed_image[i][j] != error_image[i][j]).any():
                actual += 1
    accuracy =   error/ actual * 100
    print(f'Error: {error}, Accuracy: {accuracy:.2f}%')
    io.imshow(detect_image)
    io.show()