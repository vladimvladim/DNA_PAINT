#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[2]:


values = []

def d_gaussian(x,amp1,mu1,sig1,c1,amp2,mu2,sig2,c2):
    fun_val1 = c1+amp1*np.exp(-np.power(x-mu1,2.)/(2*np.power(sig1,2.)))
    fun_val2 = c2+amp2*np.exp(-np.power(x-mu2,2.)/(2*np.power(sig2,2.)))
    return fun_val1 + fun_val2

def gaussian(x,amp,mu,sig,c):
    fun_val = c+amp*np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2.)))
    return fun_val

for i in range(1,10):
    data = pd.read_csv(str(i)+'.csv',encoding = "ISO-8859-1")
    x = data["Distance_(µm)"]
    y = data["Gray_Value"]
    
    popt,pcov=curve_fit(d_gaussian,x,y)
    
    values.append((abs(popt[1]-popt[5])*1000))


# In[4]:


#plt.plot(x,y,'o',label="Data")

print(values)


# In[17]:


#initial_guess = (250,40,5,30,300,40,5,20)
#bnd = ((30,0,0,0, 30,0,0,0),(1000,300,100,50, 1000,300,100,50))
data = pd.read_csv(str(3)+'.csv',encoding = "ISO-8859-1")
x = data["Distance_(µm)"]
y = data["Gray_Value"]

popt,pcov=curve_fit(d_gaussian,x,y)
print(popt)

px = np.linspace(x[0],x[len(x)-1],100)
yfit = d_gaussian(px,*popt)
yfit1 = gaussian(px,*popt[0:4])
yfit2 = gaussian(px,*popt[4:8])

plt.plot(x,y,'o',label="Data")
plt.plot(px,yfit,linewidth=2,label="Fit")
#plt.plot(px,yfit1,label="Peak 1")
#plt.plot(px,yfit2,label="Peak 2")

print(abs(popt[1]-popt[5])*1000," nm")


# In[26]:


# all 20 by hand

data = pd.read_csv(str(18)+'.csv',encoding = "ISO-8859-1")
x = data["Distance_(µm)"]
y = data["Gray_Value"]

#guess = (35,0.06,0.02,3,50,0.12,0.02,0)

popt,pcov=curve_fit(d_gaussian,x,y,maxfev=5000)
print(popt)

px = np.linspace(x[0],x[len(x)-1],100)
yfit = d_gaussian(px,*popt)
yfit1 = gaussian(px,*popt[0:4])
yfit2 = gaussian(px,*popt[4:8])

plt.plot(x,y,'o',label="Data")
plt.plot(px,yfit,linewidth=2,label="Fit")
plt.xlabel("Distance (μm)")
plt.ylabel("Relative brightness")
plt.legend()
plt.savefig('fit.png')
#plt.plot(px,yfit1,label="Peak 1")
#plt.plot(px,yfit2,label="Peak 2")

print(abs(popt[1]-popt[5])*1000," nm")


# In[76]:


# with guess

data = pd.read_csv(str(1)+'.csv',encoding = "ISO-8859-1")
x = data["Distance_(µm)"]
y = data["Gray_Value"]

guess = (125,0.07,0.04,25,150,0.15,0.04,25)

popt,pcov=curve_fit(d_gaussian,x,y,p0=guess,maxfev=5000)
print(popt)

px = np.linspace(x[0],x[len(x)-1],100)
yfit = d_gaussian(px,*popt)
yfit1 = gaussian(px,*popt[0:4])
yfit2 = gaussian(px,*popt[4:8])

plt.plot(x,y,'o',label="Data")
plt.plot(px,yfit,linewidth=2,label="Fit")
#plt.plot(px,yfit1,label="Peak 1")
#plt.plot(px,yfit2,label="Peak 2")

print(abs(popt[1]-popt[5])*1000," nm")


# In[87]:


meas = [79.82639358283517,61.33856748362512,50.237964176959636,59.05896177104587,37.84748561539419,54.891311260965985,\
        58.93526087234178,38.14933574823002,41.82524193871293,43.59978605907764,46.50232948072859,\
       50.31601836094917,45.609111409604814,42.26077644530357,65.5193208438155,45.31534317210554,67.4933194829379,\
       56.67707813709406,59.49227134872666,65.42943688719777]
#bad ones sorted

tot = sum(meas)    
avg = tot/len(meas)
print(tot,avg)

s = pd.Series(meas)
s.describe()


# In[ ]:




