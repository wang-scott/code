#生成的驗證碼要補0的數量
import math
def complement_zero(len_auth):
    len_zero=len_auth*(2**(7-math.floor(math.log(len_auth,2))))
    # print(math.floor(math.log(len_auth,2)))
    print(len_zero,f'須補 {len_zero-128} 個0')
    
    return len_zero

for i in range(2,8):
    print(f"Embedded length={i=}")
    complement_zero(i)