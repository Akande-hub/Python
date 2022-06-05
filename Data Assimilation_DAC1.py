#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import matplotlib.pyplot as plt


# # Question 2a

# In[2]:

#####Correct#####
def process(alpha,x0,N):
    #N is the total number of state estimates, including initial condition x0
    x= np.zeros(N)
    x[0]=x0
    for i in range(N-1):
        x[i+1]=alpha*x[i]*(1-x[i])
    return x


# In[3]:


x = process(1.2,0.2,3)


# In[4]:

#####Correct#####
def data(x,T):
    N=len(x)
    eps=np.random.normal(0,T,N)
    y=x+eps
    return y


# In[5]:

#####Correct#####

T = 0.002
Y1=[]
Y2=[]
for i in range(2000):
    Y = data(x,T)
    Y1 +=[Y[1]]
    Y2 +=[Y[2]]

sns.displot(Y1, kde=True)
plt.show()
sns.displot(Y2,kde=True)
plt.show()


# # Question b

# In[7]:


Y_array=np.array([Y1,Y2])
print(Y_array,"\n")
print("Y_mean=",Y_array.mean())


# In[8]:

#####Correct, find the mean of Y1 and Y2 as well#####

Y_covariance = np.cov(Y1, Y2)
print("Y_cov = ", "\n", Y_covariance)


# # Question c

# 
# \begin{align*}
# f_{Y|X}(y_1,y_2|x_1,x_2) &= \frac{1}{\tau \sqrt{2 \pi}} exp \lbrace \frac{-1}{2} \sum^{2}_{i=1} \frac{(y_i - x_i)^2}{\tau^2} \rbrace \\
# Max (f_{Y|X})&= Max \lbrace \frac{1}{\tau \sqrt{2 \pi}} exp \lbrace \frac{-1}{2} \sum^{2}_{i=1} \frac{(y_i - x_i)^2}{\tau^2} \rbrace \\
# &= min \lbrace \left[ \frac{(y_1 - x_1)}{\tau^2} + \frac{(y_2 - x_2)}{\tau^2}\right] \rbrace
# \end{align*}
# 
# 
#  Assuming $y_1$ and $y_2$ are independent.
# $Cov(y_1,y_2) = 0$
# \begin{align*}
# Cov(\textbf{Y}) = \begin{bmatrix}
# \tau^2 & 0\\
# 0 & \tau^2
# \end{bmatrix}
# \end{align*}
# \begin{align*}
# Cov^{-1}(\textbf{Y}) &= \frac{1}{\tau^4} \begin{bmatrix}
# \tau^2 & 0\\
# 0 & \tau^2
# \end{bmatrix}
# \end{align*}
# Therefore, we compute for $(Y-X) Cov^{-1}(Y) (Y-X)^T$,
# \begin{align*}
# (Y-X) Cov^{-1}(Y) (Y-X)^T &= \frac{1}{\tau^4} \begin{bmatrix}
# y_1-x_1 & y_2 - x_2 
# \end{bmatrix} \begin{bmatrix}
# \tau^2 & 0\\
# 0 & \tau^2
# \end{bmatrix} \begin{bmatrix}
# y_1 - x_1\\
# y_2 - x_2
# \end{bmatrix}\\
# &= \frac{1}{\tau^4} \begin{bmatrix}
# y_1 - x_1 & y_2 - x_2
# \end{bmatrix}
# \begin{bmatrix}
# \tau^2 (y_1 - x_1)\\
# \tau^2 (y_2 - x_2)
# \end{bmatrix}
# \end{align*}
# \begin{align*}
# &= \frac{\tau^2 ((y_1 - x_1)^2 + (y_2 - x_2)^2)}{\tau^4}\\
# &= \frac{(y_1 - x_1)^2 + (y_2 - x_2)^2}{\tau^2}
# \end{align*}
# Recall that, $x_{1} = \alpha x_0 (1-x_0)$ and $x_2 = \alpha x_1 (1-x_1)$ implies $x_2 = \alpha^2 x_0 ( 1 - x_0) (1- (\alpha x_0 (1-x_0)))$ 
# 
# Therefore, we have
# \begin{align*}
# \Rightarrow min \lbrace \frac{(y_1 - \alpha x_0 (1-x_0))^2 + (y_2 - \alpha^2 x_0 (1-x_0) + \alpha (\alpha x_0 (1-x_0))^2)^2}{\tau^2} \rbrace
# \end{align*}
# 

# In[9]:

#####Correct, write down the cost function formula as well.#####

def cost_fun(x0, alpha, y_1,y_2, tau):
    cost = ((y_1 - alpha*x0*(1-x0))**2 + ((y_2 - alpha**2 *x0*(1-x0) + alpha*(alpha*x0*(1-x0))**2)**2))/(tau**2)
    return cost


# In[10]:


Cost = cost_fun(0.2,1.2,Y1[1],Y2[2],0.002)
print(Cost)


# # Question d

# In[11]:

#####Correct, you iterate through the len of (x_11)#####

x_11=np.linspace(0,1,2000)
j=[]
for i in range(2000):  
    j+=[cost_fun(x_11[i],1.2,Y1[1],Y2[2], 0.002)]


# In[12]:


plt.plot(x_11,j, color = "red")
plt.ylabel("Cost function")
plt.xlabel("x_11")
plt.show()


# In[24]:


x_11[np.argmin(j)]


# # Question e
#####Correct, Compare this estimate with the least squares estimate you found in the Process model exercise.#####

# In[13]:


x0_min = x_11[np.argmin(j)]
print('Minumum occurs at x0=',x0_min)
x_0filt, x_1filt = process(1.2,x0_min,2)
print('Filtered x_1=',x_1filt)


# The re-analysis estimate satisfies $x_1 = 1.2x_0(1-x_0)$, with the value of where the
# minimum of the cost function occurs. We solve the quadratic equation $ 1.2x_0^{2} -1.2x_0 + x_1 = 0 $
# for $x_0$ .

# In[25]:


p=np.array([1.2,-1.2,x_1filt])
print(np.roots(p))


# In[ ]:





# In[ ]:




