import numpy as np
from scipy.optimize import curve_fit, leastsq

sigma = None

def func(x, b1, b2, b3):
    return (b1/b2) * np.exp(-0.5*((x-b3)/b2)**2)

data = np.loadtxt("data.dat")
x = data[:,1]
y = data[:,0]
sigma = 0.2+abs(np.random.random(size=len(x)))/5.
print sigma


p0 = (1,10,500)
popt, pcov = curve_fit(func,x,y,p0,sigma)
print "Using curve_fit"
for i in range(len(popt)):
    print popt[i], "+/-", pcov[i,i]**0.5
    
def residuals(p,y,x):
    b1, b2, b3 = p
    return (y-func(x,b1,b2,b3))/sigma

def stError(p,y,x):
    res = residuals(p,y,x)
    cst = np.dot(res,res)
    return (cst/(len(y)-len(p)))**0.5

print "Using leastsq"
plsq = leastsq(residuals, p0, args=(y, x),full_output=True)
params = plsq[0]
cov = plsq[1]
for i in range(len(params)):
    print params[i], cov[i,i]**0.5*stError(params,y,x)
