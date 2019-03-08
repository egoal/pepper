import numpy as np 
import scipy.linalg as la 

'''
utility robotic functions
'''

'''
rotations convertor

ea: Euler Angles
rm: Rotation Matrix
aa: Angle Axis
q: Quaternion, w, x, y, z
'''

def aa2q(aa):
  theta = la.norm(aa)

  return np.r_[np.cos(theta/2), aa* np.sin(theta/2)/ theta] \
    if theta> 0 else np.r_[1., aa* 0.5]

def q2aa(q):
  stheta, ctheta = la.norm(q[1:]), q[0]

  if stheta<=0:
    return q[1:]* 2. 
  else: 
    two_theta = (np.arctan2(-stheta, -ctheta) if ctheta<0 else np.arctan2(stheta, ctheta))* 2. 
    return q[1:]* two_theta/ stheta

def rm2q(rm):
  trace = np.trace(rm)
  if trace> 0.:
    t = (trace+ 1.)**0.5
    return np.r_[.5* t, np.r_[rm[2, 1]- rm[1, 2], rm[0, 2]- rm[2, 0], rm[1, 0]- rm[0, 1]]*.5/ t]
  else: 
    # max row
    i = 1 if rm[1, 1]> rm[0, 0] else 0
    i = 2 if rm[2, 2]> rm[i, i] else i 

    j = (i+1)% 3
    k = (j+1)% 3

    t = (rm[i, i]- rm[j, j]- rm[k, k]+ 1.)**.5
    return np.r_[.5*t, np.r_[rm[k, j]- rm[j, k], rm[j, i]- rm[i, j], rm[k, i]- rm[i, k]]* .5/ t]

def rm2aa(rm):
  return q2aa(rm2q(rm))

def aa2rm(aa):
  theta = la.norm(aa)
  if theta> 1e-12:
    w = aa/ theta
    c, s = np.cos(theta), np.sin(theta)
    rm = np.zeros((3, 3))
    rm[0, 0] = c+ w[0]* w[0]* (1- c)
    rm[1, 0] = w[2]* s+ w[0]* w[1]* (1-c)
    rm[2, 0] = -w[1]* s+ w[0]* w[2]* (1-c)
    rm[0, 1] = w[0]* w[1]* (1-c)- w[2]* s 
    rm[1, 1] = c+ w[1]* w[1]* (1- c)
    rm[2, 1] = w[0]* s+ w[1]* w[2]* (1-c)
    rm[0, 2] = w[1]* s+ w[0]* w[2]* (1-c)
    rm[1, 2] = -w[0]* s+ w[1]* w[2]* (1-c)
    rm[2, 2] = c+ w[2]* w[2]* (1-c)
    return rm 
  else: 
    rm = np.identity(3)
    rm[1, 0], rm[2, 0] = aa[2], -aa[1]
    rm[0, 1], rm[2, 1] = -aa[2], aa[0]
    rm[0, 2], rm[1, 2] = aa[1], -aa[0]
    return rm 

def ea2rm(ea):
  c, s = np.cos(ea), np.sin(ea)
  rm = np.zeros((3, 3))
  rm[0, :] = c[0]* c[1], -s[0]* c[2]+ c[0]* s[1]* s[2], s[0]* s[2]+ c[0]* s[1]* c[2]
  rm[1, :] = s[0]* c[1], c[0]* c[2]+ s[0]* s[1]* s[2], -c[0]* s[2]+ s[0]* s[1]* c[2]
  rm[2, :] = -s[1], c[1]* s[2], c[1]* c[2]

  return rm 

# quaternion to scaled rotation
def q2srm(q):
  aa, ab, ac, ad = q[0]* q 
  bb, bc, bd = q[1]* q[1:]
  cc, cd = q[2]* q[2:] 
  dd = q[3]* q[3]

  rm = np.zeros((3, 3))
  rm[0, :] = aa+ bb- cc- dd, 2*(bc- ad), 2* (ac+ bd)
  rm[1, :] = 2* (ad+ bc), (aa- bb+ cc- dd), 2* (cd- ab)
  rm[2, :] = 2* (bd- ac), 2* (ab+ cd), aa- bb- cc+ dd 

  return rm 

def q2rm(q):
  return q2srm(q)/ q.dot(q)

'''
rotate points
'''

def rotate_point_q(q, pt):
  t2, t3, t4 = q[0]* q[1:]
  t5, t6, t7 = -q[1]* q[1], q[1]* q[2], q[1]* q[3]
  t8, t9, t1 = -q[2]* q[2], q[2]* q[3], -q[3]* q[3]

  A = np.c_[[t8+ t1, t4+ t6, t7- t3], [t6- t4, t5+ t1, t2+ t9], 
    [t3+ t7, t9 - t2, t5+ t8]]

  return A.dot(pt)*2 + pt 

def rotate_point_aa(aa, pt):
  theta = la.norm(aa)
  if theta> 1e-12:
    c, s = np.cos(theta), np.sin(theta)
    w = aa/theta
    wcp = skew_symmetric(w).dot(pt)
    tmp = w.dot(pt)* (1- c)
    return pt* c+ wcp* s + w* tmp 
  else: 
    # R = I+ w^
    return skew_symmetric(aa).dot(pt)+ pt 

'''
'''
def quaternion_product(z, w):
  zw = np.zeros(4)
  zw[0] = z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3];
  zw[1] = z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2];
  zw[2] = z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1];
  zw[3] = z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0];
  return zw 

'''
lie
'''
def skew_symmetric(v):
  return np.c_[[0, v[2], -v[1]], [-v[2], 0, v[0]], [v[1], -v[0], 0]]

if __name__=='__main__':
  aa = np.r_[np.pi/2, 0, 0]
  p = np.r_[1, 2, 3]

  print(aa2rm(aa).dot(p))
  print(rotate_point_aa(aa, p))
