#做以下更動
#1 hong2020 unsolvablecase alpha=2
#2 proposed alpha=2 and 不管b+3是否也會是灰階值
#%%
import cv2
import numpy as np
from invert_LSB_module import *
from mainprogram import *
from skimage.metrics import structural_similarity as ssim
# names=['Lena','Tiffany','Peppers','sailboat','splash','baboon','Jet','house']
# names=['Peppers','sailboat','splash','baboon']
names=['Tiffany']
#%% 比嵌入一樣的情況
# lls=[2,3,4,5]
lls=[4]
for ll in lls:
    for name in names:      
        img = cv2.imread('./image/'+name+'.tiff')
        #各通道需要嵌入的長度
        len_r , len_b = ll,ll
        print(name,ll)

        # [論文方法]遷入驗證碼至r,b通道
        # alpha = 2
        # embedded_matrix,hong_embedded_num = hong_method(img,len_r,len_b,alpha)
        #[Proposed method]
        alpha = 2
        second_matrix, proposed_embedded_num= propose_main(img,len_r,len_b,alpha)
        #HONG2023方法
        # new_img,hong2023_embedded_num,heyhey=hong2023_main(img,len_r,len_b)

        #psnr
        # total_psnr = three_dim_psnr(img,embedded_matrix)
        # total2023_psnr =three_dim_psnr(img,new_img)
        total_second_psnr = three_dim_psnr(img,second_matrix)
        print(' Proposed:',total_second_psnr)
        print(round(ssim(img,second_matrix,multichannel=True),5))
        print('鑲嵌數量:',proposed_embedded_num)

        # 檢驗灰階值
        # gray=rgb2gray(img)
        # gray1=rgb2gray(embedded_matrix)
        # gray2=rgb2gray(new_img)
        # gray3=rgb2gray(second_matrix)
        # print('Verify:',np.sum(gray != gray1),np.sum(gray != gray2),np.sum(gray != gray3)) 


        #存檔
        # with open('./embedded/'+name+str(len_b)+'.txt', 'w') as file:
        #     for row in heyhey:
        #         row_as_text = ' '.join(map(str, row))  
        #         file.write(row_as_text + '\n')

        # cv2.imwrite('./embedded/'+'hong'+name+str(len_b)+'.tiff',embedded_matrix) #HONG
        # cv2.imwrite('./embedded/'+'new'+name+str(len_b)+'.tiff',new_img)#HONG2023
        # cv2.imwrite('./embedded/'+'proposed'+name+str(len_b)+'.tiff',second_matrix)#proposed

# %% Image quality comparisons 嵌入不平衡
lrlb =[(2,3),(3,2),(2,4),(4,2),(2,5),(3,4),(4,3),(5,2),(2,6),(3,5),(5,3),(6,2)]
names=['Jet','house']
results=[]
for lr,lb in lrlb:
    print(f'len_r={lr},len_b={lb}')
    result=[]
    for name in names:      
        img = cv2.imread('./image/'+name+'.tiff')
        len_r , len_b = lr,lb       
        print(name,len_r , len_b)
        #[Proposed method]
        alpha = 2
        second_matrix, proposed_embedded_num= propose_main(img,len_r,len_b,alpha)       
        proposed_psnr = three_dim_psnr(img,second_matrix)##PSNR
        print(f'Proposed={proposed_psnr}')
        result.append(round(proposed_psnr,2))
    results.append(result)

        

# %%
print(names)
for i in range(len(results)):
    print(lrlb[i],results[i])

# %%
