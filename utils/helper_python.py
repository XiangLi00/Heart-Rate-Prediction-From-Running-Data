from datetime import datetime, timedelta
import os
import sys
import time

import numpy as np

# Interpoliert [a,b] geometrisch mit n Werten
def rangee(a,b,n):
  if n==1:
    return [a]
  x=np.zeros(n)
  factor=(b/a)**(1/(n-1))
  x[0]=a
  for i in range(1,n):
    x[i]=x[i-1]*factor
  return x  

# Interpoliert [a,b] geometrisch mit n Werten. Auf int gerundet
def rangeeInt(a,b,n):
    if n==1:
        return [a]
    x=[0]*n
    factor=(b/a)**(1/(n-1))
    x[0]=a
    for i in range(1,n):
        x[i]=int((factor**i)*a)
    return x  