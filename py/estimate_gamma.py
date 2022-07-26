#!/usr/bin/python3 

import numpy as np 

'''
estimate a gamma distribution Thomas P Minka
'''

# todo: handle parameters
def estimate(x):
  from scipy.special import digamma, polygamma

  avg_x = np.mean(x)
  avg_log_x = np.mean(np.log(x))
  log_avg_x = np.log(avg_x)

  a = 0.5/ (log_avg_x- avg_log_x)
  for i in range(4):
    dinva = (avg_log_x- log_avg_x+ np.log(a)- digamma(a))/ (a**2* (1/a- polygamma(1, a)))
    a = 1/(1/a + dinva)

  b = avg_x/ a

  return (a, b)

if __name__=='__main__':
  x = np.loadtxt('data/chi2.txt')
  print(estimate(x**0.5))
