from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

def psnr(img1,img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    else:
        return 20*np.log10(255/np.sqrt(mse))


pic = ['Lena','Tiffany','Sailboat','Baboon','Peppers','Splash']
pic = ['house','Jet']
name=['hong','new','proposed']
for x in pic:
    img = cv2.imread('./image/'+x+'.tiff')
    print(x)
    for i in name:
        print(i)
        img2 = cv2.imread('./embedded/'+i+x+'2.tiff')
        img3 = cv2.imread('./embedded/'+i+x+'3.tiff')
        img4 = cv2.imread('./embedded/'+i+x+'4.tiff')

        print(round(psnr(img,img2),5),round(ssim(img,img2,multichannel=True),5))
        print(round(psnr(img,img3),5),round(ssim(img,img3,multichannel=True),5))
        print(round(psnr(img,img4),5),round(ssim(img,img4,multichannel=True),5))


    