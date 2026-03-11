import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from dataclasses import dataclass
from dataclasses import replace

#constatnts
dr=1.0e-3
GRID_POINTS=1001
c=1.0
m=1.0
@dataclass
class Parameters:
    n:float=1.0
    K:float=1.0e2
    RHO_C:float=1.28e-3

p=Parameters()

def boundary_conditions(r,Rs,M,p):
    '''
    경계조건을 설정하는 함수
    '''
    if r==0:
        h=p.K*(p.n+1)*p.RHO_C**(1/p.n)
        dh=0
        dphi=0
        ddphi=4.0/3.0*np.pi*p.RHO_C*Rs**2
    elif r==1:
        h=0
        dh=-M/Rs
        dphi=M/Rs
        ddphi=-2.0*M/Rs
    else:
        print("Error: r should be 0 or 1")
    return np.array([h, dh, dphi, ddphi])

def deriv(r,y,Rs,M,p):
    '''
    미분을 계산하는 함수
    '''
    h,dphi=y
    if (r==0 or r==1):
        return boundary_conditions(r,Rs,M,p)[1::2]
    else:
        dh=-dphi
        par=max(h,0)/(p.K*(p.n+1))
        ddphi=(-2.0/r)*dphi+4.0*np.pi*Rs**2*par**p.n
    return np.array([dh, ddphi])


def RK2_step(r,y,Rs,M,p,dr):
    '''
    RK2의 다음 스텝을 계산하는 함수
    y는 [h,dphi/dr]의 배열
    '''
    k1=deriv(r,y,Rs,M,p)
    k2=deriv(r+dr,y+dr*k1,Rs,M,p)
    ynext=y+dr*(k1+k2)/2
    return ynext


def shoot(Rs,M,p):
    '''
    f(R_s,M)=h_(+)-h_(-)
    g(R_s,M)=dPhi_(+)/dr - dPhi_(-)/dr
    의 값을 반환하는 함수
    '''
    hp,hm,dphip,dphim=np.zeros(501),np.zeros(501),np.zeros(501),np.zeros(501)
    hp[0],dphip[0]=boundary_conditions(1,Rs,M,p)[::2]
    hm[0],dphim[0]=boundary_conditions(0,Rs,M,p)[::2]
    for i in range(1,501):
        rp=1-i*dr
        rm=i*dr
        hp[i],dphip[i]=RK2_step(rp,np.array([hp[i-1],dphip[i-1]]),Rs,M,p,-dr)
        hm[i],dphim[i]=RK2_step(rm,np.array([hm[i-1],dphim[i-1]]),Rs,M,p,dr)
    f=hp[500]-hm[500]
    g=dphip[500]-dphim[500]
    return f,g

def Jacobian(Rs, M, p, eps=1e-6):
    
    f_p, g_p = shoot(Rs + eps, M, p)
    f_m, g_m = shoot(Rs - eps, M, p)
    df_dRs = (f_p - f_m) / (2.0 * eps)
    dg_dRs = (g_p - g_m) / (2.0 * eps)

    
    f_p, g_p = shoot(Rs, M + eps, p)
    f_m, g_m = shoot(Rs, M - eps, p)
    df_dM  = (f_p - f_m) / (2.0 * eps)
    dg_dM  = (g_p - g_m) / (2.0 * eps)

    J = np.array([[df_dRs, df_dM],
                  [dg_dRs, dg_dM]], dtype=float)
    return J



def find_Rs_M(Rs,M,p,error=10e-12, roop = 1000, eps=1e-6):
    '''
    뉴턴-랩슨법으로
    R_s와 M을 찾는 함수
    10e-12의 오차범위
    '''
    x = np.array([Rs,M],dtype=float)
    for k in range(roop) :
        f,g = shoot(x[0],x[1],p)
        F = np.array([f,g],dtype=float)
        if k%100==0:
            print(f"error at iteration {k}: {np.linalg.norm(F)}")
        if np.linalg.norm(F) < error :
            return x[0],x[1]
            
        J = Jacobian(x[0],x[1],p,eps)
    
        dx = np.linalg.solve(J,-F)
    
        x = x + dx

        new_Rs = x[0] + dx[0]
        new_M = x[1] + dx[1]
        
        if new_Rs <= 0: new_Rs = 0.1
        if new_M <= 0: new_M = 0.1
    print("반복 횟수 초과")
    return x[0], x[1] 

def get_h(Rs,M,p):
    '''
    최종 h값 구하는 함수
    '''
    r_values=np.linspace(0,1,GRID_POINTS)
    h_values=np.zeros(GRID_POINTS)
    h_values[0],dphi=boundary_conditions(0,Rs,M,p)[::2]
    for i in range(1,GRID_POINTS):
        y=np.array([h_values[i-1],dphi])
        h_values[i],dphi=RK2_step(r_values[i-1],y,Rs,M,p,dr)
    return h_values

def dat_output(p):
    '''
    결과를 dat파일로 저장하는 함수
    '''
    f=open('problem1.dat','w')
    Rs,M=find_Rs_M(15.0,1.0,p)
    f.write(f"{p.n:.3f} {M:.8f} {Rs:.8f}\n")
    r_values=np.linspace(0,1,GRID_POINTS)
    h_values=get_h(Rs,M,p)
    for i in range(GRID_POINTS):
        f.write(f"{r_values[i]:.3f} {max(0,h_values[i]):.8f}\n")
    print("결과가 'problem1.dat'로 저장되었습니다.")
    f.close()

def plot_graph2(p):
    '''
    1-(2)번 문제를 그래프로 출력하는 함수
    '''
    n=[0.8,1.0,1.5]
    r_values=np.linspace(0,1,GRID_POINTS)
    for i in n:
        p=replace(p,n=i)
        Rs,M=find_Rs_M(15.0,1.4,p)
        h_values=get_h(Rs,M,p)
        density=(np.maximum(h_values,0)/(p.K*(p.n+1)))**p.n
        plt.plot(r_values,density,label=f"n={i}")
    plt.xlabel(r"$\hat{r}$")
    plt.ylabel(r"$\rho$")
    plt.legend()
    plt.grid(True)
    plt.title("1(2)")
    plt.savefig('problem1-2.png')
    print("그래프가 'problem1-2.png'로 저장되었습니다.")

def plot_graph3(p):
    '''
    1-(3)번 문제를 그래프로 출력하는 함수
    
    '''
    n_value=[0.8,1.0,1.5]
    K_value=[500,50,5]
    num_points=50
    rhoc_value=np.logspace(-3,0,num_points)
    plt.figure()
    for i in range(3):
        Rs_list=[]
        M_list=[]
        curr_Rs, curr_M = 10.0, 0.5
        p=replace(p,n=n_value[i],K=K_value[i])
        print(f"\n--- n={n_value[i]}, K={K_value[i]} 계산 시작 ---")
        for j in range(num_points):
            p=replace(p,RHO_C=rhoc_value[j])
            Rs,M=find_Rs_M(curr_Rs,curr_M,p)
            escape_velocity=np.sqrt(2*M/Rs)
            print(f"n={n_value[i]}, K={K_value[i]}, rho_c={rhoc_value[j]:.2e} -> Rs={Rs:.4f}, M={M:.4f}, v_esc={escape_velocity:.4f}")
            if escape_velocity>=c:
                break
            Rs_list.append(Rs)
            M_list.append(M)
            curr_Rs, curr_M = Rs, M
        Rs_list=np.array(Rs_list)
        M_list=np.array(M_list)
        sort_idx=np.argsort(Rs_list)
        plt.plot(Rs_list[sort_idx],M_list[sort_idx],label=f"n={n_value[i]}, K={K_value[i]}")

    plt.xlabel(r"$R_s$")
    plt.ylabel(r"$M$")
    plt.legend()
    plt.grid(True)
    plt.title("1(3)")
    plt.savefig('problem1-3.png')
    print("그래프가 'problem1-3.png'로 저장되었습니다.")

def main():
    dat_output(p=replace(p,n=1.5))#dat 파일의 n값 설정(n=?)
    
if __name__ == "__main__":
    main()