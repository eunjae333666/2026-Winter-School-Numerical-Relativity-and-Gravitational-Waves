import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dataclasses import replace
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import problem1 #problem1.py 파일 먼저 실행해야함

#constatnts
dr=1.0e-3
GRID_POINTS=1001
c=1.0
m=1.0

@dataclass
class Parameters:
    n:float=np.sqrt(3)
    K:float=3.00
    RHO_C:float=1.28e-3
    Rs:float=None
    M:float=None
    h0:object=None
    dh0:object=None
    def __post_init__(self):
        p1_params = problem1.Parameters(n=self.n, K=self.K, RHO_C=self.RHO_C)
        self.Rs, self.M = problem1.find_Rs_M(15.0, 1.0, p1_params)
        self.h0_origin, self.dphi_origin=problem1.get_h_dphi(self.Rs,self.M,self)
        self.dh0_origin=-self.dphi_origin
        self.h0=interp1d(np.linspace(0,1,GRID_POINTS),self.h0_origin,kind='cubic')
        self.dh0=interp1d(np.linspace(0,1,GRID_POINTS),self.dh0_origin,kind='cubic')


p=Parameters()

def boundary_conditions(r,w,p):
    '''
    경계조건을 설정하는 함수
    '''
    if r==0:
        ksi=1.0
        dksi=0.0
        delh=-3.0*p.h0(0.0)*ksi/(p.n*p.Rs)
        ddelh=0.0
    elif r==1:
        ddh0=-2.0*p.dh0(1.0) #문제 1의 두 식에서 유도됨

        ksi=1.0
        dksi=-1.0/(p.n+1)/p.dh0(1.0)*(3.0*p.dh0(1.0)*ksi+p.n*p.Rs*p.Rs*w*w+p.n*ddh0*ksi)
        delh=-1.0/p.Rs*p.dh0(1.0)*ksi
        ddelh=p.Rs*w*w*ksi
    else:
        print("Error: r should be 0 or 1")
    return np.array([ksi, dksi, delh, ddelh])

def deriv(r,y,w,p):
    '''
    미분을 계산하는 함수
    '''
    ksi,delh=y
    if (r==0 or r==1):
        return boundary_conditions(r,w,p)[1::2]
    else:
        ddelh=p.Rs*w*w*r*ksi
        dksi=-3.0*ksi/r-p.n*p.Rs/p.h0(r)/r*delh-ksi*p.n*p.dh0(r)/p.h0(r)
    return np.array([dksi, ddelh])


def RK2_step(r,y,w,p,dr):
    '''
    RK2의 다음 스텝을 계산하는 함수
    y는 [ksi,delh]의 배열
    '''
    k1=deriv(r,y,w,p)
    k2=deriv(r+dr,y+dr*k1,w,p)
    ynext=y+dr*(k1+k2)/2
    return ynext


def shoot(w,p):
    '''
    f=ksi_core*delh_surface - ksi_surface*delh_core
    의 값을 반환하는 함수
    '''
    ksip, ksim, delhp, delhm=np.zeros(501),np.zeros(501),np.zeros(501),np.zeros(501)
    ksip[0],delhp[0]=boundary_conditions(1,w,p)[::2]
    ksim[0],delhm[0]=boundary_conditions(0,w,p)[::2]
    for i in range(1,501):
        rp=1-i*dr
        rm=i*dr
        ksip[i],delhp[i]=RK2_step(rp,np.array([ksip[i-1],delhp[i-1]]),w,p,-dr)
        ksim[i],delhm[i]=RK2_step(rm,np.array([ksim[i-1],delhm[i-1]]),w,p,dr)
    f=ksip[500]*delhm[500]-ksim[500]*delhp[500]
    return f

def find_w(p, w1 = 0.001, w2 = 1.0, root = 3, dw = 0.001) :
    """
    w를 찾는 함수 
    구간 w1, w2 사이에서 0되는 w를 찾기
    brentq 함수 : 구간 사이에서 0을 찾음(단, f(w1)*f(w2)<0 이어야 함.)
    """
    modes = []
    
    w_list = np.arange(w1, w2, dw)
    f_start = shoot(w_list[0],p)
    
    for i in range(1,len(w_list)) :
        f_current = shoot(w_list[i],p)
        
        if f_start * f_current < 0 : 
            w_root = brentq(lambda w : shoot(w,p), w_list[i-1], w_list[i])

            modes.append(w_root)
            if i%100==0: print(f"Found mode at w = {w_root}")
            
            if len(modes) == root :
                break
        f_start = f_current
    
    return modes

def get_ksi_delh_two(w,p):
    '''
    양쪽에서 적분하여 최종 ksi와 delh값 구하는 함수
    '''
    r_values=np.linspace(0,1,GRID_POINTS)
    ksi_values=np.zeros(GRID_POINTS)
    delh_values=np.zeros(GRID_POINTS)

    mid = GRID_POINTS // 2 - 1
    
    ksi_values[0],delh_values[0]=boundary_conditions(0,w,p)[::2]
    for i in range(1,mid+1):
        y=np.array([ksi_values[i-1],delh_values[i-1]])
        ksi_values[i],delh_values[i]=RK2_step(r_values[i-1],y,w,p,dr)
    ksi_values[-1],delh_values[-1]=boundary_conditions(1,w,p)[::2]
    for i in range(1,GRID_POINTS-mid-1):
        r_rev=r_values[-i]
        y=np.array([ksi_values[-i],delh_values[-i]])
        ksi_values[-i-1],delh_values[-i-1]=RK2_step(r_rev,y,w,p,-dr)

    A = ksi_values[mid+1] / ksi_values[mid] 
    
    ksi_values[:mid+1] *= A
    delh_values[:mid+1] *= A

    return ksi_values, delh_values


def dat_output(p,mode=0):
    '''
    결과를 dat파일로 저장하는 함수
    '''
    f=open('problem2.dat','w')
    w=find_w(p)
    f.write(f"{p.M:.8f} {p.Rs:.8f} {w[mode]}\n")
    r_values=np.linspace(0,1,GRID_POINTS)
    ksi_value,delh_value=get_ksi_delh_two(w[mode],p)
    for i in range(GRID_POINTS):
        f.write(f"{r_values[i]:.3f} {ksi_value[i]:.8f} {delh_value[i]:.8f}\n")
    print("결과가 'problem2.dat'로 저장되었습니다.")
    f.close()

def plot_graph(p):
    '''
    2-(3)번 문제를 그래프로 출력하는 함수
    '''
    r_values=np.linspace(0,1,GRID_POINTS)
    w=find_w(p)
    ksi_list=[]
    delh_list=[]
    for i in w:
        ksi_values,delh_values=get_ksi_delh_two(i,p)
        ksi_list.append(ksi_values)
        delh_list.append(delh_values)
    
    labels=["fundamental","1st overtone","2nd overtone","3rd overtone"]

    #ksi 그래프
    plt.figure()
    for i in range(len(w)):
        lbl=labels[i] if i<len(labels) else f"{i}th overtone"
        plt.plot(r_values,ksi_list[i],label=lbl)
        for j in range(0,GRID_POINTS,10):
           print(f"w={w[i]:.6f}, r={r_values[j]:.3f}, ksi={ksi_list[i][j]:.6f}")
    plt.xlabel(r"$\hat{r}$")
    plt.ylabel(r"$\hat{\xi}$")
    plt.legend()
    plt.grid(True)
    plt.title(r"2(3) $\hat{\xi}$ vs $\hat{r}$")
    plt.savefig('problem2-3ksi.png')
    print("그래프가 'problem2-3ksi.png'로 저장되었습니다.")
    plt.close()

    #delh 그래프
    plt.figure()
    for i in range(len(w)):
        lbl=labels[i] if i<len(labels) else f"{i}th overtone"
        plt.plot(r_values,delh_list[i],label=lbl)
    plt.xlabel(r"$\hat{r}$")
    plt.ylabel(r"$\delta h$")
    plt.legend()
    plt.grid(True)
    plt.title(r"2(3) $\delta h$ vs $\hat{r}$")
    plt.savefig('problem2-3delh.png')
    print("그래프가 'problem2-3delh.png'로 저장되었습니다.")
    plt.close()

def main():
    #dat 파일의 모드 번호 설정(mode=?)
    #0: fundamental mode
    #1: 1st overtone
    #...
    dat_output(p,mode=1)
    
if __name__ == "__main__":
    main()