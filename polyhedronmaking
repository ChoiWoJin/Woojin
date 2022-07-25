from sympy import symbols, Matrix, solve
import matplotlib.pyplot as plt
import numpy as np
import math

#drawing을 위한 초기설정
fig = plt.figure(figsize = (9,6))
ax = fig.add_subplot(1,1,1, projection = '3d')
iter_max=200

#두점간 geodesic
def draw_geodesic(a1,b1,c1,a2,b2,c2,m):
    r1 = 0; r2 = 0
    if a1 == a2 and b1 == b2:
        h3 = np.linspace(min(c1, c2), max(c1, c2), m)
        h1 = np.linspace(a1, a1, m)
        h2 = np.linspace(b1, b1, m)
        ax.plot(h1,h2,h3)
     
    else :
        r1 = ((c2**2-c1**2)*(a2-a1))/(2*(a2-a1)**2 + 2*(b2-b1)**2)+(a1+a2)/2
        h1 = np.linspace(min(a1, a2), max(a1, a2), m)
        if a2 != a1:
            r2 = ((b2-b1)/(a2-a1))*(r1-a1)+b1
            h2 = ((b2-b1)/(a2-a1))*(h1-a1)+b1
        else:
            r2 = (c2**2-c1**2)/(2*(b2-b1))+(b1+b2)/2
            h2 = np.linspace(min(b1, b2), max(b1, b2), m)
        R = np.linspace(math.sqrt((r1-a1)**2 + (r2-b1)**2 + c1**2),math.sqrt((r1-a1)**2 + (r2-b1)**2 + c1**2),m)
        h3 = (R**2 - (h1-r1)**2 - (h2-r2)**2)**(1/2)
        ax.plot(h1,h2,h3)

#로바쳅스키 -> upper model    
def upper_model(x,y,z,w):
    if x != y:
        return (-z/(x+y),-w/(x+y),1/(x+y))

#초기 설정(손으로.)
v1 = [0,0,0,1]; v2 = [0,0,1,0]; v3 = [0,1,0,0]; v4 = [1,-3,-2,-1]; v5 = [1,-1,-2,-3]; v6 = [2,-1,-2,-3]
V = Matrix([v1,v2,v3,v4,v5,v6])
length = 5
G = Matrix([[-4, 0, 0, 0],[0, 2, -1, 0],[0, -1, 2, -1],[0, 0, -1, 2]])


#기저변환
P, D = G.diagonalize(); P = P.T
for i in range(4):
    Norm = 0
    for j in range(4):
        Norm = Norm + P[i,j]**2
    Norm = math.sqrt(Norm)
    for j in range(4):
        P[i,j] = P[i,j]/Norm

pp = 0
for i in range(4):
    if D[i,i] < 0:
        pp = i
    D[i,i] = math.sqrt(abs(D[i,i]))

P = (P.T)*D
if pp != 0:
    for i in range(4):
        if i != pp:
            mk = 0
            mk = P[0,pp];P[0,pp] = P[0,i];P[0,i] = mk
            mk = P[1,pp];P[1,pp] = P[1,i];P[1,i] = mk 
            mk = P[2,pp];P[2,pp] = P[2,i];P[2,i] = mk 
            mk = P[3,pp];P[3,pp] = P[3,i];P[3,i] = mk       
            break

k1 = (float(P[0,0]),float(P[0,1]),float(P[0,2]),float(P[0,3]))
k2 = (float(P[1,0]),float(P[1,1]),float(P[1,2]),float(P[1,3]))
k3 = (float(P[2,0]),float(P[2,1]),float(P[2,2]),float(P[2,3]))
k4 = (float(P[3,0]),float(P[3,1]),float(P[3,2]),float(P[3,3]))

def elvector(a,b,c,d,mm):
    v_0 = a*k1[0] + b*k2[0] + c*k3[0] + d*k4[0]
    v_1 = a*k1[1] + b*k2[1] + c*k3[1] + d*k4[1]
    v_2 = a*k1[2] + b*k2[2] + c*k3[2] + d*k4[2]
    v_3 = a*k1[3] + b*k2[3] + c*k3[3] + d*k4[3]
    if mm == 1:
        K = math.sqrt(v_0**2 - v_1**2 - v_2**2 - v_3**2)
        if v_0 > 0:
            return ((v_0)/K,(v_1)/K,(v_2)/K,(v_3)/K)
        else:
            return (-(v_0)/K,-(v_1)/K,-(v_2)/K,-(v_3)/K)
    elif mm == 0:
        if v_0 > 0:
            v_0 = v_0 + 0.00005
            K = math.sqrt(v_0**2 - v_1**2 - v_2**2 - v_3**2)
            return ((v_0)/K,(v_1)/K,(v_2)/K,(v_3)/K)
        else:
            v_0 = v_0 - 0.00005
            K = math.sqrt(v_0**2 - v_1**2 - v_2**2 - v_3**2)
            return (-(v_0)/K,-(v_1)/K,-(v_2)/K,-(v_3)/K)

#Find edges
Edges = []
for i in range(length):
    for j in range(i+1,length):
        K = Matrix([V.row(i),V.row(j)])
        VV = K*G*K.T
        if float(VV.det()) > 0:
            Edges.append([i+1,j+1])
print(Edges)

EliptVer = []
InftyVer = []
Ver=[]

#Find simple vertex
for i in range(length):
    for j in range(i+1,length):
        for k in range(j+1,length):
            K = Matrix([V.row(i),V.row(j),V.row(k)])
            VV = K*G*K.T
            if float(VV.det()) > 0:
                KK = Matrix([V.row(i),V.row(j)])
                VVV = KK*G*KK.T
                if float(VVV.det()) > 0:
                    EliptVer.append([i+1,j+1,k+1])
                    Ver.append([i+1,j+1,k+1])
            elif float(VV.det()) == 0:
                KK1 = Matrix([V.row(i),V.row(j)])
                VV1 = KK1*G*KK1.T
                KK2 = Matrix([V.row(i),V.row(k)])
                VV2 = KK2*G*KK2.T
                KK3 = Matrix([V.row(j),V.row(k)])
                VV3 = KK3*G*KK3.T
                if float(VV1.det()) > 0 and float(VV2.det()) > 0 and float(VV3.det()) > 0:
                    InftyVer.append([i+1,j+1,k+1])
                    Ver.append([i+1,j+1,k+1])

#Find non-simple vertex
for i in range(length):
    for j in range(i+1,length):
        for k in range(j+1,length):
            for p in range(k+1,length):        
                K1 = Matrix([V.row(i),V.row(j)])
                V1 = K1*G*K1.T
                if float(V1.det()) == 0:
                    KK1 = Matrix([V.row(k),V.row(p)])
                    VV1 = KK1*G*KK1.T
                    VK1 = K1*G*KK1.T
                    if float(VV1.det()) == 0 and float(VK1[0,0]) == 0 and float(VK1[0,1]) == 0 and float(VK1[1,1]) == 0 and float(VK1[0,0]) == 0:
                        InftyVer.append([i+1,j+1,k+1,p+1])
                        Ver.append([i+1,j+1,k+1,p+1])
                        break
                K2 = Matrix([V.row(i),V.row(k)])
                V2 = K2*G*K2.T
                if float(V2.det()) == 0:
                    KK2 = Matrix([V.row(j),V.row(p)])
                    VV2 = KK2*G*KK2.T
                    VK2 = K2*G*KK2.T
                    if float(VV2.det()) == 0 and float(VK2[0,0]) == 0 and float(VK2[0,1]) == 0 and float(VK2[1,1]) == 0 and float(VK2[0,0]) == 0:
                        InftyVer.append([i+1,j+1,k+1,p+1])
                        Ver.append([i+1,j+1,k+1,p+1])
                        break
                K3 = Matrix([V.row(i),V.row(p)])
                V3 = K3*G*K3.T
                if float(V3.det()) == 0:
                    KK3 = Matrix([V.row(j),V.row(k)])
                    VV3 = KK3*G*KK3.T
                    VK3 = K3*G*KK3.T
                    if float(VV3.det()) == 0 and float(VK3[0,0]) == 0 and float(VK3[0,1]) == 0 and float(VK3[1,1]) == 0 and float(VK3[0,0]) == 0:
                        InftyVer.append([i+1,j+1,k+1,p+1])
                        Ver.append([i+1,j+1,k+1,p+1])
                        break
                                
print(EliptVer)
print(InftyVer)
print(Ver)

#Determine whether such polytope have a finite volume
for l in range(len(Edges)):
    t = 0
    for k in range(len(Ver)):
        if Edges[l][0] in Ver[k] and Edges[l][1] in Ver[k]: 
            t = t+1
    if t != 2:
        print("Not finite")
        quit()
print("Finite")
                
#Drawing polytope
for l in range(len(Edges)):
    H = (0,0,0,0); K = (0,0,0,0)
    Dver=[]
    for k in range(len(Ver)):
        if Edges[l][0] in Ver[k] and Edges[l][1] in Ver[k]: 
            Dver.append(Ver[k])
    for m in range(2):
        x,y,z,w = symbols("x y z w")
        if len(Dver[m]) == 3:
            B = Matrix([V.row(Dver[m][0]-1),V.row(Dver[m][1]-1),V.row(Dver[m][2]-1)])
            for n in range(4):
                if n == 0:
                    X = Matrix([[1,y,z,w]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(y,z,w))
                    if solution:
                        (a,b,c,d) = (1,float(solution[y]),float(solution[z]),float(solution[w]))
                        break
                elif n == 1:
                    X = Matrix([[x,1,z,w]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(x,z,w))
                    if solution:
                        (a,b,c,d) = (float(solution[x]),1,float(solution[z]),float(solution[w]))
                        break 
                elif n == 2:
                    X = Matrix([[x,y,1,w]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(x,y,w))
                    if solution:
                        (a,b,c,d) = (float(solution[x]),float(solution[y]),1,float(solution[w]))
                        break
                elif n == 3:
                    X = Matrix([[x,y,z,1]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(x,y,z))
                    if solution:
                        (a,b,c,d) = (float(solution[x]),float(solution[y]),float(solution[z],1))
                        break
            
            if Dver[m] in InftyVer:
                if m == 0:
                    H = elvector(a,b,c,d,0)
                else:
                    K = elvector(a,b,c,d,0) 
            else:
                if m == 0:
                    H = elvector(a,b,c,d,1)
                else:
                    K = elvector(a,b,c,d,1)    
        else:
            B = Matrix([V.row(Dver[m][0]-1),V.row(Dver[m][1]-1),V.row(Dver[m][2]-1),V.row(Dver[m][3]-1)])
            for n in range(4):
                if n == 0:
                    X = Matrix([[1,y,z,w]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(y,z,w))
                    if solution:
                        (a,b,c,d) = (1,float(solution[y]),float(solution[z]),float(solution[w]))
                        break
                elif n == 1:
                    X = Matrix([[x,1,z,w]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(x,z,w))
                    if solution:
                        (a,b,c,d) = (float(solution[x]),1,float(solution[z]),float(solution[w]))
                        break 
                elif n == 2:
                    X = Matrix([[x,y,1,w]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(x,y,w))
                    if solution:
                        (a,b,c,d) = (float(solution[x]),float(solution[y]),1,float(solution[w]))
                        break
                elif n == 3:
                    X = Matrix([[x,y,z,1]])
                    L = B*G*X.T
                    AEq=[]
                    for i in range(3):
                        AEq.append(L[i,0])
                    solution = solve(tuple(AEq),(x,y,z))
                    if solution:
                        (a,b,c,d) = (float(solution[x]),float(solution[y]),float(solution[z],1))
                        break
            if m == 0:
                H = elvector(a,b,c,d,0)
            else:
                K = elvector(a,b,c,d,0)      
    
                  
    (a1,b1,c1) = upper_model(round(H[0],4),round(H[1],4),round(H[2],4),round(H[3],4))
    (a2,b2,c2) = upper_model(round(K[0],4),round(K[1],4),round(K[2],4),round(K[3],4))
    draw_geodesic(a1,b1,c1,a2,b2,c2,iter_max)
 
plt.show()
