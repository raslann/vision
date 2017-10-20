import numpy as np
import scipy as sp
import scipy.linalg

world = np.loadtxt('world.txt')

image = np.loadtxt('image.txt')

world = np.concatenate([world, np.ones((1, 10))])

image = np.concatenate([image, np.ones((1, 10))])

A = np.zeros((0, 12))

for i in range (10):

    x = image [ : , i]
    X = world [ : , i]
    a1 = np.concatenate([np.zeros(4), -x[2] * X, x[1] * X]).reshape(1, -1)
    a2 = np.concatenate([x[2] * X, np.zeros(4), -x[0] * X]).reshape(1, -1)
    A = np.concatenate([A, a1, a2])


u, d, v = np.linalg.svd(A)
p = v[d.argmin()].reshape((3, 4))

print('P is: ', p)

# verify re-projection
image_p = p.dot(world)
image_p = image_p / image_p[2]
print ('re-projection', image_p)


u, d, v = np.linalg.svd(p)
c = v[3]
print ('C is:', c)

k, r = sp.linalg.rq(p, mode='economic')

r_ = r [:, -1]
r_ = r [:, :-1]
t= r[:, -1]
c2 = np.linalg.solve(r_, -t)
print('verified C is:', c2) # [ 1. -1. -1.]



