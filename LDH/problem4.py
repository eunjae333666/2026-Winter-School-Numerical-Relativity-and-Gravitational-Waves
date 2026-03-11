import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dataclasses import replace

#constatnts
dr=1.0e-3
GRID_POINTS=1001
c=1.0

@dataclass
class Parameters:
    n:float=1.0
    K:float=1.0e2
    RHO_C:float=1.28e-3

p=Parameters()

def boundary_conditions(r,Rs,lambdas,Phic,p):
    '''
    경계조건을 설정하는 함수
    '''
    hc=2.0 #임의의 값
    if r==0:
        lambda0=lambdas
        dlambda0=0.0
        Phi0=-Phic
        dPhi0=0.0
        h0=hc
        dh0=0.0
        
    elif r==1:
        lambda0=lambdas
        dlambda0=-(1-np.exp(2*lambdas))/2
        Phi0=-lambdas
        dPhi0=(1-np.exp(2*lambdas))/2
        h0=1.0
        dh0=-(1-np.exp(2*lambdas))/2
    else:
        print("Error: r should be 0 or 1")
    return np.array([lambda0,dlambda0,Phi0,dPhi0,h0, dh0])

def deriv(r,y,Rs,lambdas,Phic,p):
    '''
    미분을 계산하는 함수
    '''
    lambda0,Phi0,h0=y

    if (r==0 or r==1):
        return boundary_conditions(r,Rs,lambdas,Phic,p)[1::2]
    else:
        dlambda=(1-np.exp(2*lambda0))/2/r + 4.0*np.pi*r*Rs*Rs*np.exp(2*lambda0)*(((max(h0,0)-1)/(p.K*(p.n+1)))**p.n+(h0-1)/(p.n+1)*p.n)
        dPhi=-(1-np.exp(2*lambda0))/2/r + 4.0*np.pi*r*Rs*Rs*np.exp(2*lambda0)*(max(h0,0)-1)/(p.n+1)
        dh=-h0*dPhi
    return np.array([dlambda,dPhi,dh])


def RK2_step(r,y,Rs,lambdas,Phic,p,dr):
    '''
    RK2의 다음 스텝을 계산하는 함수
    y는 [h,dphi/dr]의 배열
    '''
    k1=deriv(r,y,Rs,lambdas,Phic,p)
    k2=deriv(r+dr,y+dr*k1,Rs,lambdas,Phic,p)
    ynext=y+dr*(k1+k2)/2
    return ynext


def shoot(Rs,lambdas,Phic,p):
    '''
    f(R_s,M)=h_(+)-h_(-)
    g(R_s,M)=dPhi_(+)/dr - dPhi_(-)/dr
    의 값을 반환하는 함수
    '''
    lambdap,lambdam,Phip,Phim,hp,hm=np.zeros(501),np.zeros(501),np.zeros(501),np.zeros(501),np.zeros(501),np.zeros(501)
    lambdap[0],Phip[0],hp[0]=boundary_conditions(1,Rs,lambdas,Phic,p)[::2]
    lambdam[0],Phim[0],hm[0]=boundary_conditions(0,Rs,lambdas,Phic,p)[::2]
    for i in range(1,501):
        rp=1-i*dr
        rm=i*dr
        lambdap[i],Phip[i],hp[i]=RK2_step(rp,np.array([lambdap[i-1],Phip[i-1],hp[i-1]]),Rs,lambdas,Phic,p,-dr)
        lambdam[i],Phim[i],hm[i]=RK2_step(rm,np.array([lambdam[i-1],Phim[i-1],hm[i-1]]),Rs,lambdas,Phic,p,dr)
    f=hp[500]-hm[500]
    g=lambdap[500]-lambdam[500]
    t=Phip[500]-Phim[500]
    return f,g,t

def Jacobian(Rs,lambdas,Phic, p, eps=1e-6):
    
    f_p, g_p, h_p = shoot(Rs + eps, lambdas, Phic, p)
    f_m, g_m, h_m = shoot(Rs - eps, lambdas, Phic, p)
    df_dRs = (f_p - f_m) / (2.0 * eps)
    dg_dRs = (g_p - g_m) / (2.0 * eps)
    dh_dRs = (h_p - h_m) / (2.0 * eps)
    
    f_p, g_p, h_p = shoot(Rs, lambdas + eps, Phic, p)
    f_m, g_m, h_m = shoot(Rs, lambdas - eps, Phic, p)
    df_dlambdas = (f_p - f_m) / (2.0 * eps)
    dg_dlambdas = (g_p - g_m) / (2.0 * eps)
    dh_dlambdas = (h_p - h_m) / (2.0 * eps)
    
    f_p, g_p, h_p = shoot(Rs, lambdas, Phic + eps, p)
    f_m, g_m, h_m = shoot(Rs, lambdas, Phic - eps, p)
    df_Phic = (f_p - f_m) / (2.0 * eps)
    dg_Phic = (g_p - g_m) / (2.0 * eps)
    dh_Phic = (h_p - h_m) / (2.0 * eps)
    
    J = np.array([df_dRs, df_dlambdas, df_Phic],
                 [dg_dRs, dg_dlambdas, dg_Phic],
                 [dh_dRs, dh_dlambdas, dh_Phic], dtype = float)
    
    return J



def find_Rs_lambdas_Phic(Rs,lambdas,Phic,p,error=10e-12, roop = 1000, eps=1e-6):
    '''
    뉴턴-랩슨법으로
    R_s와 lambda_s, Phi_c을 찾는 함수
    10e-12의 오차범위
    '''
    x = np.array([Rs,lambdas,Phic],dtype=float)
    for k in range(roop) :
        f,g,t = shoot(x[0],x[1],x[2],p)
        F = np.array([f,g,t],dtype=float)
        if np.linalg.norm(F) < error :
            return x[0],x[1],x[2]
            
        J = Jacobian(x[0],x[1],x[2],p,eps)
    
        dx = np.linalg.solve(J,-F)
    
        x = x + dx
        '''
        필요한지 잘 모르겠음.
        new_Rs = x[0] + dx[0]
        new_lambdas = x[1] + dx[1]
        new_Phic = x[2] + dx[2]

        if new_Rs <= 0: new_Rs = 0.1
        if new_M <= 0: new_M = 0.1
        '''
    print("반복 횟수 초과")
    return x[0], x[1], x[2]

def get_lambda0_Phi0_h0(Rs,lambdas,Phic,p):
    '''
    최종 lambda0, Phi0, h0 값 구하는 함수
    '''
    r_values=np.linspace(0,1,GRID_POINTS)
    lambda0_values=np.zeros(GRID_POINTS)
    Phi0_value=np.zeros(GRID_POINTS)
    h0_value=np.zeros(GRID_POINTS)
    lambda0_values[0],Phi0_value[0],h0_value[0]=boundary_conditions(0,Rs,lambdas,Phic,p)[::2]
    for i in range(1,GRID_POINTS):
        y=np.array([lambda0_values[i-1],Phi0_value[i-1],h0_value[i-1]])
        lambda0_values[i],Phi0_value[i],h0_value[i]=RK2_step(r_values[i-1],y,Rs,M,p,dr)
    return lambda0_values,Phi0_value,h0_value

def dat_output(p):
    '''
    결과를 dat파일로 저장하는 함수
    '''
    f=open('problem4.dat','w')
    Rs,lambdas,Phic=find_Rs_lambdas_Phic(15.0,1.0,p)
    f.write(f"{p.n:.3f} {Rs:.8f} {lambdas:.8f} {Phic:.8f}\n")
    r_values=np.linspace(0,1,GRID_POINTS)
    lambda0,Phi0,h0=get_lambda0_Phi0_h0(Rs,M,p)
    for i in range(GRID_POINTS):
        f.write(f"{r_values[i]:.3f} {lambda0[i]:.8f} {Phi0[i]:.8f} {np.maximum(h0,0):.8f}\n")
    print("결과가 'problem4.dat'로 저장되었습니다.")
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
        h_values,dphi_values=get_h_dphi(Rs,M,p)
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
    rhoc_value=np.logspace(-5,0,num_points)
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