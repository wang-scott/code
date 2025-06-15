#%%
import cv2
import numpy as np
from invert_LSB_module import *
from mainprogram import *
from skimage.metrics import structural_similarity as ssim
names=['Peppers','sailboat','splash','baboon','Jet','house']


#%%
# lls=[2,3,4,5]
lls=[4]
for ll in lls:
    for name in names:      
        print(name,len_r , len_b)
        img = cv2.imread('./image/'+name+'.tiff')
        gray_img = rgb2gray(img)
        bin_matrix = dec2bin(img)
        auth_code = hash_all_pixel(img,len_r,len_b) 
        hong_img = zhou_img = proposed_img =np.zeros((img.shape)) 
        #各通道需要嵌入的長度
        len_r , len_b = ll,ll

        #找出HONG的unsolvable grayscale value
        alpha = 2
        for i in range(bin_matrix.shape[0]):
            for j in range(bin_matrix.shape[1]):
                hong_img[i][j][2] = int(Embedding(bin_matrix[i][j][2],auth_code[i][j][:len_r],length=len_r),2)#red
                hong_img[i][j][0] = int(Embedding(bin_matrix[i][j][0],auth_code[i][j][len_r:],length=len_b),2)#blue
        u_g_v = cal_green(hong_img,gray_img)
        print(u_g_v)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                ii = i*gray_img.shape[1]+j+1
                
                if gray_img[i][j] in u_g_v: #unsolvable case
                    #====HONG
                    hong_img[i][j]=generate_perturbed_pairs(img[i][j],ii,alpha)
                    #====ZHOU
                    zhou_img[i][j] = CRS(img[i][j],ii)
                    #====proposed (只使用IACDC)
                    proposed_img[i][j] = zhou_img[i][j]

                else : #一般情況
                    #====HONG
                    hong_img[i][j][1]=round((gray_img[i][j]-0.299*hong_img[i][j][2]-0.114*hong_img[i][j][0])/0.587)
                    #====ZHOU
                    if MSBA(img[i][j],len_r,len_b,ii)!= None:
                        zhou_img[i][j] = MSBA(img[i][j],len_r,len_b,ii)
                    else:
                        print("ZHOU:ERROR "+"i="+i+"j="+j)
                    #====proposed (只使用IACDC)
                    a = int(Embedding(bin_matrix[i][j][2],auth_code[i][j][:len_r],length=len_r),2)#red
                    b = int(Embedding(bin_matrix[i][j][2],auth_code[i][j][:len_r],1,length=len_r),2)#red
                    proposed_img[i][j][2] = eee(a,b,img[i][j][2])
                    c = int(Embedding(bin_matrix[i][j][0],auth_code[i][j][len_r:],length=len_b),2)#blue
                    d = int(Embedding(bin_matrix[i][j][0],auth_code[i][j][len_r:],1,length=len_b),2)#blue
                    proposed_img[i][j][0] = eee(c,d,img[i][j][0])  
                    proposed_img[i][j][1]=round((gray_img[i][j]-0.299*proposed_img[i][j][2]-0.114*proposed_img[i][j][0])/0.587)
                
                
                
                #proposed (只使用IACDC)

# %%
show(hong_img.astype(np.uint8))
# %%
