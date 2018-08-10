import numpy as np
import sys
import math
from scipy.special import erf

g = int(sys.argv[1]) # real space gridpoints
g3 = g**3 # total gridpoints
space = float(sys.argv[2]) #
p = np.linspace(-space,space,g)
h = p[1]-p[0]
#one_d = -space:space:g*j
gj = g*1j
[X,Y,Z] = np.mgrid[-space:space:gj,-space:space:gj,-space:space:gj]
xs = np.ravel(X)
ys = np.ravel(Y)
zs = np.ravel(Z)
R = (xs**2. + ys**2. + zs**2.)**(0.5)
Vext = -2./R
e = np.linspace(1,1,g)
ncomp = np.exp(-R**2./2.)
ncomp = -2.*ncomp/sum(ncomp) / h**3.
ncomppot = -2./R*erf(R/math.sqrt(2.))
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import eigsh, cgs
L = spdiags([e,-2*e,e],[-1,0,1],g,g)/h**2
I = eye(g,g)
L3 = kron(kron(L,I),I) + kron(kron(I,L),I) + kron(kron(I,I),L)
Vtot = Vext
#ncomp = 
tol = 1e-3
print 'Iter'.rjust(6),'Eigenvalue'.rjust(10),'KE'.rjust(8),'Exch. E'.rjust(8),'Ext. E'.rjust(8),'Pot. E.'.rjust(8),'E_tot'.rjust(8),'diff'.rjust(8)
print '----'.rjust(6),'----------'.rjust(10),'----'.rjust(8),'-------'.rjust(8),'------'.rjust(8),'------'.rjust(8),'-----'.rjust(8),'-----'.rjust(8)
count = 1
diff = 1.
pi = 3.14159
Elast = 0.
const = 27.21
while abs(diff) > tol:
    # Subpsace diagnolizations subroutine
    if count>1:
        H = -0.5*L3+spdiags(Vtot,0,g3,g3)
        RR = H*psi - E*psi
        psie = np.array([np.ravel(psi), np.ravel(RR)])
        HH = np.dot(psie,H*np.transpose(psie))
        SS = np.dot(psie,np.transpose(psie))
        [E, U] = eigsh(HH,M=SS,k=1)
        psie = np.dot(np.transpose(psie),U)
        print np.dot(np.transpose(psie),H*psi)
    E,psi = eigsh(-0.5*L3+spdiags(Vtot,0,g3,g3),k=1,which='SA')
    #if count==2:
        #break
    psi = psi / h**(1.5)
    n = 2.*psi**(2.)
    Vx = -(3./3.14)**(1./3.)*n**(1./3.)
    Vh = cgs(L3, - 4*pi*(np.ravel(n)+ncomp), tol = 1e-7,maxiter = 400)[0] - ncomppot
    #break
    Vtot = np.ravel(Vx) + Vh + Vext
    T = 2.*sum(np.dot(np.ravel(psi),(-0.5*L3)*psi))*h**3
    Eext = np.dot(np.ravel(n),Vext)*h**3
    Eh = 0.5*np.dot(np.ravel(n),Vh)*h**3
    Ex = sum(sum((-3./4.)*(3./pi)**(1./3.)*n**(4./3.)))*h**3.
    Etot = T + Eext + Eh + Ex
    diff = Etot - Elast
    Elast = Etot
    print str(count).rjust(6),('%.2f'%np.real(E[0]*const)).rjust(10), ('%.2f'%np.real(T*const)).rjust(8),('%.2f'%np.real(Ex*const)).rjust(8),('%.2f'%np.real(Eext*const)).rjust(8),('%.2f'%np.real(Eh*const)).rjust(8),('%.3f'%np.real(Etot*const)).rjust(8),('%.3f'%np.real(diff*const)).rjust(8)
    count+=1
    #break
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot()
#fig.savefig('out.png'))
