import numpy as np
import scipy
from timeit import default_timer as timer

# Count Sketch
def genCountSketch(a,b,k):
	S = np.zeros((a,b))
	for i in range(b):
	    S[np.random.choice(a,1)[0]][i] = 2*np.random.choice(2, 1)[0] - 1
	return S


def genPHD(a,b,k):
	P = np.zeros((a,b))
	H = scipy.linalg.hadamard(b)
	D = np.zeros((b,b))
	for i in range(b):
	    D[i][i] = 2*np.random.choice(2, 1)[0] - 1
	for i in range(a):
	    P[i][np.random.choice(a,1)[0]] = 1
	return np.dot(P, np.dot(H, D))

def genGaussian(a,b,k):
	return numpy.sqrt(1.0/k) * np.random.randn(a, b)

n = 10000
d = 100
# r = 1000
# k = 50
A = np.random.rand(n,d)


pairs = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
funcs = [genCountSketch,genPHD,genGaussian]
names = ["genCountSketch","genPHD","genGaussian"]
for p in pairs:
	print "Testing %s and %s" % (names[0],names[1])
	func1 = funcs[p[0]]
	func2 = funcs[p[1]]
	for k in [50,60,70,80,90]:
		for r in [500,1000,1500]:
			print "n=%d d=%d k=%d r=%d" % (n, d, k, r)
			S = func1(r,n,k)
			R = func2(d,r,k)

			start = timer()
			SA = np.matmul(S,A)
			SAR = np.matmul(SA,R)
			AR = np.matmul(A,R)
			SARpsuedo = np.linalg.pinv(SAR)

			T = np.matmul(np.matmul(AR,SARpsuedo),SAR)
			U, S, V = np.linalg.svd(T, full_matrices=False)
			end = timer()
			
			print "Sketching method: %f s" % (end - start)

			Tk = np.zeros((len(U), len(V)))
			for i in xrange(k):
			    Tk += S[i] * np.outer(U.T[i], V[i])

			out = np.matmul(np.matmul(Tk,SARpsuedo),SA)

			start = timer()
			U2, S2, V2 = np.linalg.svd(A, full_matrices=False)
			end = timer()
			print "Base method: %f s" % (end - start)
			Ak = np.zeros((len(U2), len(V2)))
			for i in xrange(k):
			    Ak += S2[i] * np.outer(U2.T[i], V2[i])

			base = np.linalg.norm(A - Ak)
			us = np.linalg.norm(A - out)
			print "Base norm: %f, Sketching Norm: %f" % (base, us)

