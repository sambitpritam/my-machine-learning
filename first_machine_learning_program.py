# -*- coding: utf-8 -*-

# import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import os

def plt_model(x, y, models, fname, mx=None, ymax=None, xmin=None):
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web Traffice over last Month")
    plt.xlabel("Time")
    plt.ylabel("Hits/Hour")
    plt.xticks([w*7*24 for w in range(10)], ['week %d' % w for w in range(10)])
    
    if models:
        if mx is None:
            mx = np.linspace(0, x[-1], 1000)
            
        for model, style, color in zip(models, linestyles, colors):
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, color=color)
            
        plt.legend(["d=%i" % m.order for m in models], loc="upper left")
        
    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    
    if ymax:
        plt.ylim(ymax=ymax)
    
    if xmin:
        plt.xlim(xmin=xmin)
        
    plt.grid(True, linestyle="-", color="0.75")
    plt.savefig(fname)

b = np.genfromtxt('web_traffic.tsv', delimiter='\t')
colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

print(b[:10])
print('shape: ', b.shape)

# PREPROCESSING AND CLEANING OF DATA

# segregating the entire data set into 2 data structures for easy data handling
x = b[:, 0]
y = b[:, 1]

# check how many invalid values are there
print('Invalid values in x: ', np.sum(np.isnan(x)))
print('invalid values in y: ', np.sum(np.isnan(y)))

# as we have 8 invalid values in y, we can choose to ignore it 
# because it is 1% of the entire data set
# lets remove the invalid columns and clean the data set
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

print('Invalid values in y after cleaning: ', np.sum(np.isnan(y)))

# Using first model- a straight line (order 1)
# In order to use any of the model, the model should be able to calculate the 
# Error as below
def error(f, x, y):
    return np.sum((f(x)-y)**2)


fp1, residuals, rank, sp, rcond = np.polyfit(x, y, 1, full=True)
print("Model Parameters: %s" % fp1)
print("Residuals: %s" % residuals)

# using poly1d to create the model function
# f1 = sp.poly1d(fp1)
f1 = np.poly1d(fp1)
err = error(f1, x, y)

print(type(err))

fx = np.linspace(0, x[-1], 1000)

# trying with an advance model with degree=2
fp2 = np.polyfit(x, y, 2)
f2 = np.poly1d(fp2)
print("Approximation Error : %s" % error(f2, x, y))

# trying with advanced model degree=3
fp3 = np.polyfit(x, y, 3)
f3 = np.poly1d(fp3)
print("Approximation Error for Degreee 3 : %s" % error(f3, x, y))

# trying with advance model degree=10
fp10 = np.polyfit(x, y, 10)
f10 = np.poly1d(fp10)
print("Approximation Error for Degree 10 : %s" % error(f10, x, y))

# trying with advance model degree=100
fp100 = np.polyfit(x, y, 100)
f100 = np.poly1d(fp100)
print("Approximation Error for Degree 100 : %s" % error(f100, x, y))

# -------------------------------------------------------
# Taking a step back
# -------------------------------------------------------
# as we can see there is an inflection between week 3 and 4
# lets break the data into 2 parts
# data before week 3.5 and post 3.5
inflection = 3.5*7*24 # converting week 3.5 into hours
print("Inflection hour for 3.5 is : %f" % inflection)
inflection = int(inflection)
# breaking the data into 2 parts
# data before week 3.5
xa = x[:inflection]
ya = y[:inflection]

# data post week 3.5
xb = x[inflection:]
yb = y[inflection:]

# calculating polynomial coefficient for both the data points for degree = 1
fa = np.poly1d(np.polyfit(xa, ya, 1))
fb = np.poly1d(np.polyfit(xb, yb, 1))

# calculating approximation error for both data set
error_a = error(fa, xa, ya)
error_b = error(fb, xb, yb)

print("Error Inflection: %f" % (error_a + error_b))

plt_model(x, y, [fa, fb], os.path.join("..","15-03-2018_2.png"))

# as the data set 2 depicts a better picture of the future data, 
# lets apply various degree of polynomial coefficient to the data set 2
fb2 = np.poly1d(np.polyfit(xb, yb, 2))
fb3 = np.poly1d(np.polyfit(xb, yb, 3))
fb10 = np.poly1d(np.polyfit(xb, yb, 10))
fb100 = np.poly1d(np.polyfit(xb, yb, 100))

# Approximation errors for data after inflection
for f in [fb, fb2, fb3, fb10, fb100]:
    print("Inflection error for %i is %f" % (f.order, error(f, xb, yb)))
    
plt_model(x, y, [fb, fb2, fb3, fb10, fb100], os.path.join("./results", "15-03-2018_3.png"), mx=np.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# separating test data and training data
frac = 0.3
split_index = int(frac * len(xb))
shuffled = np.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_index])
train = sorted(shuffled[split_index:])

# Lets train our model
fbtr1 = np.poly1d(np.polyfit(xb[train], yb[train], 1))
fbtr2 = np.poly1d(np.polyfit(xb[train], yb[train], 2))
fbtr3 = np.poly1d(np.polyfit(xb[train], yb[train], 3))
fbtr10 = np.poly1d(np.polyfit(xb[train], yb[train], 10))
fbtr100 = np.poly1d(np.polyfit(xb[train], yb[train], 100))

# Calculating the approximation error on the test data set using the models created using train data set
for f in [fbtr1, fbtr2, fbtr3, fbtr10, fbtr100]:
    print("Approximation Error for %i is %f" % (f.order, error(f, xb[test], yb[test])))
    
plt_model(x, y, [fbtr1, fbtr2, fbtr3, fbtr10, fbtr100], os.path.join("./results", "10-03-2018_4.png")
,mx=np.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)


# =============================================================================
# 
# # plotting the current dataset on matplot lib
# plt.scatter(x, y, s=10)
# plt.title('Web traffice over the last month')
# plt.xlabel('time')
# plt.ylabel('Hits/hour')
# plt.xticks([w*7*24 for w in range(10)], ['week %d' % w for w in range(10)])
# plt.autoscale(tight=True)
# #to draw a slightly opaque and dashed lines
# plt.grid(True, linestyle='-', color='0.75')
# # plot for degree=1 model
# plt.plot(fx, f1(fx), linewidth=3, color="green")
# plt.legend(["d=%i" % f1.order], loc="best")
# 
# # plot for degree=2 model
# plt.plot(fx, f2(fx), linewidth=3, color="black")
# plt.legend(["d=%i" % f2.order], loc="best")
# 
# #  plot degree=3 model
# plt.plot(fx, f3(fx), linewidth=3, color="red")
# plt.legend(["d=%i" % f3.order], loc="best")
# 
# # plot for degree=10 model
# plt.plot(fx, f10(fx), linewidth=3, color="yellow")
# plt.legend(["d=%i" % f10.order], loc="best")
# 
# # plot for degree=100 model
# plt.plot(fx, f100(fx), linewidth=3, color="cyan")
# plt.legend(["d=%i" % f100.order], loc="best")
# 
# plt.show()
# 
# # ----------------------------------------
# # for split data set
# # ----------------------------------------
# plt.scatter(x, y, s=10)
# plt.title('Web traffice over the last month')
# plt.xlabel('time')
# plt.ylabel('Hits/hour')
# plt.xticks([w*7*24 for w in range(10)], ['week %d' % w for w in range(10)])
# plt.autoscale(tight=True)
# #to draw a slightly opaque and dashed lines
# plt.grid(True, linestyle='-', color='0.75')
# 
# # plot for polynomial Coefficient of degree 1
# plt.plot(fx, fa(fx), linewidth=3, color="green")
# plt.plot(fx, fb(fx), linewidth=3, color="blue")
# 
# plt.show()
# =============================================================================



    


