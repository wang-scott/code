#%%
import cv2
import numpy as np
from invert_LSB_module import *
name = 'baboon'


# %% Proposed驗證
ll = 4
len_r,len_b=ll,ll
alpha=2
origin_path = './image/'+name+'.tiff'
embedded_path = './embedded/proposed'+name+str(len_b)+'.tiff'
# path = './embedded/proposed'+name+str(len_b)+'.tiff'
path = './tamper/proposed'+name+'tamper_0.7.tif'

origin = cv2.imread(origin_path)
embeddded = cv2.imread(embedded_path)
img = cv2.imread(path)

groundtruth = check_tamper_num(embeddded,img)

def UUU(img,len_r,len_b):
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
def proposed_detect_tamper(origin,img,alpha,len_r,len_b):
    U_second,U_second_un = UUU(origin,len_r,len_b)
    # U_second=[]
    # U_second_un=[]
 
    bin_matrix = dec2bin(img)
    gray_img = rgb2gray(img)
    origin_auth_code = hash_all_pixel(origin,len_r,len_b) #測試用
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
    return embedded_tamper_matrix
find_tamper = proposed_detect_tamper(origin,img,alpha,len_r,len_b)     

show(find_tamper)
show(groundtruth)
cv2.imwrite('./tamper/'+name+'groundtruth.tiff',groundtruth)
# cv2.imwrite('./tamper/'+name+'detect'+'.tiff',find_tamper)

output = np.zeros_like(find_tamper, dtype=np.uint8)
condition1 = (groundtruth == 255) & (np.all(find_tamper == 255, axis=-1))
condition1_3d = np.repeat(condition1[:, :, np.newaxis], 3, axis=2)
output[condition1_3d] = 255
show(output)
cv2.imwrite('./tamper/'+name+'detect'+'.tiff',output)
compare_with_groundtruth(groundtruth,output)


# %%論文檢測
embedded_path = './embedded/hong'+name+str(len_b)+'.tiff'
path = './tamper/hong'+name+'tamper_0.7.tif'

embeddded = cv2.imread(embedded_path)
img = cv2.imread(path)

hong_groundtruth = check_tamper_num(embeddded,img)

def aka_unsolvable(img,len_r,len_b):
    embedded_matrix = np.zeros((img.shape))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    bin_matrix = dec2bin(img)
    authentication_code = hash_all_pixel(img,len_r,len_b) 
    print(embedded_matrix.shape)
    for i in range(bin_matrix.shape[0]):
        for j in range(bin_matrix.shape[1]):
            embedded_matrix[i][j][2] = int(Embedding(bin_matrix[i][j][2],authentication_code[i][j][:len_r],length=len_r),2)#red
            embedded_matrix[i][j][0] = int(Embedding(bin_matrix[i][j][0],authentication_code[i][j][len_r:],length=len_b),2)#blue
    return cal_green(embedded_matrix,gray_img)
def HONG_detect(origin,img,len_r,len_b):
    alpha=1
    hong_tamper = np.zeros((img.shape)).astype(np.uint8)
    U = aka_unsolvable(origin,len_r,len_b)
    bin_matrix = dec2bin(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#用其灰階值
    new_auth_code = hash_all_pixel(img,len_r,len_b)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gray_img[i][j] not in U:
                rr = int(Embedding(bin_matrix[i][j][2],new_auth_code[i][j][:len_r],length=len_r),2)#red
                bb = int(Embedding(bin_matrix[i][j][0],new_auth_code[i][j][len_r:],length=len_b),2)#blue
                gg = round((gray_img[i][j]-0.299*rr-0.114*bb)/0.587)
                if img[i][j][0]!=bb or img[i][j][2]!=rr:
                    hong_tamper[i][j]=255               
            else:
                ii = i*gray_img.shape[1]+j+1
                tt = generate_perturbed_pairs(img[i][j],ii,alpha)!= img[i][j]
                if True in tt :
                    hong_tamper[i][j]=255            
    return hong_tamper
    
hong_find_tamper = HONG_detect(origin,img,4,4)   
cv2.imwrite('./tamper/'+'Hong'+name+'detect'+'.tiff',hong_find_tamper)
compare_with_groundtruth(hong_groundtruth,hong_find_tamper)


# %% HONG2023
len_r,len_b =4,4
img = cv2.imread('./embedded/new'+name+str(len_b)+'.tiff')
tamper = cv2.imread('./tamper/new'+name+'tamper_0.7.tif')

groundtruth = check_tamper_num(img,tamper)

with open('./embedded/baboon4.txt', 'r') as file: #使用CRS方法藏入的位置
    lines = file.readlines()  
    data = [[int(num) for num in line.split()] for line in lines] 
    
def HONG2023_detect(tamper,data,len_r,len_b):
    check_array = np.zeros((tamper.shape))
    for i in range(tamper.shape[0]):
        if i%5 == 0:
            print(i)
        for j in range(tamper.shape[1]):
            ii =  i*tamper.shape[1]+j+1
            if [i,j] in data:
                bb,gg,rr=CRS(tamper[i][j],ii)
                if bb!=tamper[i][j][0] or gg!=tamper[i][j][1] or  rr!=tamper[i][j][2]:
                    check_array[i][j]=255
            else:
                result1 = MSBA(tamper[i][j],len_r,len_b,ii)
                if result1 != None:
                    bb,gg,rr=result1
                    if bb!=tamper[i][j][0] or gg!=tamper[i][j][1] or  rr!=tamper[i][j][2]:
                        check_array[i][j]=255
                else:
                    check_array[i][j]=255
                    # result2 = PSS(tamper[i][j],len_r,len_b,ii)
                    # if result2 !=None:
                    #     bb,gg,rr=result2
                    #     if bb!=tamper[i][j][0] or gg!=tamper[i][j][1] or  rr!=tamper[i][j][2]:
                    #         check_array[i][j]=255
                    # else:
                    #     check_array[i][j]=255
    check_array=check_array.astype(np.uint8)
    return check_array
find_2023 = HONG2023_detect(tamper,data,len_r,len_b)
show(find_2023)                

output3 = np.zeros_like(find_2023, dtype=np.uint8)
condition1 = (groundtruth == 255) & (np.all(find_2023 == 255, axis=-1))
condition1_3d = np.repeat(condition1[:, :, np.newaxis], 3, axis=2)
output3[condition1_3d] = 255
show(output3)
compare_with_groundtruth(groundtruth,output3)
cv2.imwrite('./tamper/new'+name+'detect.tiff',output3)#HONG2023



# %%
