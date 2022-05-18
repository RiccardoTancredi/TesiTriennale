import numpy as np

def x_WLC_f(f, L):
    KBT = 4.11 # pN*nm
    P = 1.35 # nm
    fnorm = ((4*P)/KBT)*f
    a2 = (1/4)*(-9-fnorm)
    a1 = (3/2)+(1/2)*fnorm
    a0 = -fnorm/4
    
    R = (9*a1*a2-27*a0-2*a2**3)/54.
    Q = (3*a1-a2**2)/9.
    
    D = Q**3+R**2
    
    if D > 0:
        # In this case, there is only one real root, given by "out" below
        S = np.cbrt(R+np.sqrt(D))
        T = np.cbrt(R-np.sqrt(D))
        out = (-1/3)*a2+S+T
    elif D < 0:
        # In this case there 3 real distinct solutions, given by out1,
        # out2, out3 below. The one that interests us is that in the
        # inerval [0,1]. It is seen ("empirically") that is always the
        # second one in the list below [there is perhaps more to search here]
        
        theta = np.arccos(R/np.sqrt(-Q**3))
        # out1 = 2*np.sqrt(-Q)*np.cos(theta/3)-(1/3)*a2;
        out2 = 2*np.sqrt(-Q)*np.cos((theta+2*np.pi)/3)-(1/3)*a2
        # out3 = 2*np.sqrt(-Q)*np.cos((theta+4*np.pi)/3)-(1/3)*a2
        
        # We implement the following check just to be sure out2 is the good root 
        # (in case this "empirical" truth turns out to stop working) 
        try:
            out2 < 0 or out2 > 1
        except:    
            print('The default root doesn"t seem the be good one - you may want to check if the others lie in the interval [0,1]')
        else:
            out = out2
    else:
        # In theory we always go from D>0 to D<0 by passing to a D=0
        # boundary, where we have two real roots (and where the formulas
        # above change again slightly). In practice, however, due to round-off errors,
        # it seems we never hit this boundary but always pass "through" it 
        # This D=0 scenario could still be implemented if needed, though.
        print('#ToDo')

    z = out
    
    return z*L