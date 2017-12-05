
# coding: utf-8

# # Using grid search, we are aiming to find optimal values for parameters U, m, r, c
# we vary So = [0.5U, 0.6U, 0.7U, 0.8U, 0.9U]

# In[2]:

import numpy as np
#from sympy import *
from sympy import *
from sympy.abc import x
from sklearn.grid_search import ParameterGrid
r = Symbol('r')
d = Symbol('d')

def phi(y):
    return integrate(exp(-x**2 /2)/sqrt(2*pi), (x, -oo, y))

def F_r(r, d):
    return (1 - 2*phi(-r/d) - 2*(1 - exp(-(r/d)**2 /2))/sqrt(2*pi)*d/r)

def rho(m, S_0, U, c, R):
    return (log(F_r(R, sqrt(1 + m/4 - 2*S_0 + U**(2**m + 1))))/ log(F_r(R, sqrt(1 + m/4 - 2*c*S_0))))



U = np.arange(0.75, 0.95, 0.05).tolist()
m = [2, 3, 4, 5]
r = np.arange(1.5, 3, 0.1).tolist()
c = np.arange(0.0, 1.0, 0.1).tolist()
param_grid = {'param1':U, 'param2':r, 'param3':m}
grid = ParameterGrid(param_grid)
S = [0.5, 0.6, 0.7, 0.8, 0.9]
val_S = []
#val_1 = []
plot_1 = []
Mp=1
Up = 0.05
Rp = 0.1
for p_ in c:
    val_1 = []
    Plot_Val=[]
    for s0 in S:
        fval = 100
        for m0 in m:
            for U0 in U:
                for r0 in r:
                    if (1 + m0/4 - 2*s0 + U0**(2**m0 + 1)) >= 0 and (1 + m0/4 - 2*p_*s0) >= 0:  
                        fval1 = rho(m0, s0, U0, p_, r0)
                    else:
                        continue
                    if(fval1 < fval):
                        #print("H")
                        Mp = m0
                        Up = U0
                        Rp = r0
                        fval = fval1
        val_1.append(fval)
        Plot_Val.append([Mp, Up, Rp])
    val_S.append(val_1)
    plot_1.append(Plot_Val)


# In[3]:

print(val_S )


# In[6]:

type(val_S[0][0])


# In[7]:

P = val_S[0][0]


# In[10]:

print P


# In[11]:

val_S[0][0].doit()


# In[14]:

import sympy
pi = sympy.pi
log = sympy.log
sqrt = sympy.sqrt


# In[15]:

val_S[0][0].doit()


# In[16]:

plot_1


# In[36]:

x = val_S[1][1]
xevaled = []
for i in range(len(val_S)):
    temp =[]
    for j in range(len(val_S[i])):
        temp.append(val_S[i][j].evalf())
    xevaled.append(temp)
print len(val_S)
print len(xevaled)


# In[37]:

xevaled


# In[38]:

len(plot_1)


# In[39]:

print(plot_1)


# In[40]:

len(plot_1[0])


# In[44]:

len(xevaled[0])


# In[46]:

S0 = []
S1 = []
S2 = []
S3 = []
S4 = []

for i in range(len(xevaled)):
    j = 0
    S0.append(xevaled[i][j])
    j = 1
    S1.append(xevaled[i][j])
    j = 2
    S2.append(xevaled[i][j])
    j = 3
    S3.append(xevaled[i][j])
    j = 4
    S4.append(xevaled[i][j])


# In[61]:

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
get_ipython().magic(u'matplotlib inline')


# In[62]:

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
objects = ['1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1']
# objects = ('L=2, K=1','','','','L=32, K=1','L=2, K=2','','','','L=32, K=2','L=2, K=4','','','','L=32, K= 4','L=2, K=8','','','','L=32, K=8')

y_pos = np.arange(len(objects))
plt.xticks(y_pos, objects)
plt.ylabel('Rho *')
plt.xlabel('c')
S0.reverse()
S1.reverse()
S2.reverse()
S3.reverse()
S4.reverse()
ax1, = plt.plot(S0,label='So = 0.5U')
ax2, = plt.plot(S1,label='So = 0.6U')
ax3, = plt.plot(S2,label='So = 0.7U')
ax4, = plt.plot(S3,label='So = 0.8U')
ax5, = plt.plot(S4,label='So = 0.9U')
plt.legend(handler_map={ax1: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax2: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax3: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax4: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax5: HandlerLine2D(numpoints=1)})


# In[64]:

len(plot_1)


# In[69]:

print plot_1[0]


# In[92]:

S0_1 = []
S0_2 = []
S0_3 = []
S1_1 = []
S1_2 = []
S1_3 = []
S2_1 = []
S2_2 = []
S2_3 = []
S3_1 = []
S3_2 = []
S3_3 = []
S4_1 = []
S4_2 = []
S4_3 = []

def fun(fun_name,num):
    fun_name.append(num)

for i in range(len(plot_1)):
    for j in range(len(plot_1[i])):
        if(j == 0):
            k = 0
            S0_1.append(plot_1[i][j][k])
            k = 1
            S0_2.append(plot_1[i][j][k])
            k = 2
            S0_3.append(plot_1[i][j][k])
        if(j == 1):
            k = 0
            S1_1.append(plot_1[i][j][k])
            k = 1
            S1_2.append(plot_1[i][j][k])
            k = 2
            S1_3.append(plot_1[i][j][k])
        if(j == 2):
            k = 0
            S2_1.append(plot_1[i][j][k])
            k = 1
            S2_2.append(plot_1[i][j][k])
            k = 2
            S2_3.append(plot_1[i][j][k])
        if(j == 3):
            k = 0
            S3_1.append(plot_1[i][j][k])
            k = 1
            S3_2.append(plot_1[i][j][k])
            k = 2
            S3_3.append(plot_1[i][j][k])
        if(j == 4):
            k = 0
            S4_1.append(plot_1[i][j][k])
            k = 1
            S4_2.append(plot_1[i][j][k])
            k = 2
            S4_3.append(plot_1[i][j][k])


# In[93]:

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
objects = ['1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1']
# objects = ('L=2, K=1','','','','L=32, K=1','L=2, K=2','','','','L=32, K=2','L=2, K=4','','','','L=32, K= 4','L=2, K=8','','','','L=32, K=8')

y_pos = np.arange(len(objects))
plt.xticks(y_pos, objects)
plt.ylabel('Opt m')
plt.xlabel('c')
S0_1.reverse()
S1_1.reverse()
S2_1.reverse()
S3_1.reverse()
S4_1.reverse()
ax1, = plt.plot(S0_1,label='So = 0.5U')
ax2, = plt.plot(S1_1,label='So = 0.6U')
ax3, = plt.plot(S2_1,label='So = 0.7U')
ax4, = plt.plot(S3_1,label='So = 0.8U')
ax5, = plt.plot(S4_1,label='So = 0.9U')
plt.legend(handler_map={ax1: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax2: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax3: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax4: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax5: HandlerLine2D(numpoints=1)})


# In[94]:

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
objects = ['1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1']
# objects = ('L=2, K=1','','','','L=32, K=1','L=2, K=2','','','','L=32, K=2','L=2, K=4','','','','L=32, K= 4','L=2, K=8','','','','L=32, K=8')

y_pos = np.arange(len(objects))
plt.xticks(y_pos, objects)
plt.ylabel('Opt m')
plt.xlabel('c')
S0_2.reverse()
S1_2.reverse()
S2_2.reverse()
S3_2.reverse()
S4_2.reverse()
ax1, = plt.plot(S0_2,label='So = 0.5U')
ax2, = plt.plot(S1_2,label='So = 0.6U')
ax3, = plt.plot(S2_2,label='So = 0.7U')
ax4, = plt.plot(S3_2,label='So = 0.8U')
ax5, = plt.plot(S4_2,label='So = 0.9U')
plt.legend(handler_map={ax1: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax2: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax3: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax4: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax5: HandlerLine2D(numpoints=1)})


# In[95]:

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
objects = ['1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1']
# objects = ('L=2, K=1','','','','L=32, K=1','L=2, K=2','','','','L=32, K=2','L=2, K=4','','','','L=32, K= 4','L=2, K=8','','','','L=32, K=8')

y_pos = np.arange(len(objects))
plt.xticks(y_pos, objects)
plt.ylabel('Opt m')
plt.xlabel('c')
S0_3.reverse()
S1_3.reverse()
S2_3.reverse()
S3_3.reverse()
S4_3.reverse()
ax1, = plt.plot(S0_3,label='So = 0.5U')
ax2, = plt.plot(S1_3,label='So = 0.6U')
ax3, = plt.plot(S2_3,label='So = 0.7U')
ax4, = plt.plot(S3_3,label='So = 0.8U')
ax5, = plt.plot(S4_3,label='So = 0.9U')
plt.legend(handler_map={ax1: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax2: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax3: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax4: HandlerLine2D(numpoints=1)})
plt.legend(handler_map={ax5: HandlerLine2D(numpoints=1)})


# In[ ]:



