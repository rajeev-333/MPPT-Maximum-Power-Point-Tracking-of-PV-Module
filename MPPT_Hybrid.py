import numpy as np
import matplotlib.pyplot as plt

# ==============================
# PV PARAMETERS
# ==============================
Isc = 5.0
Voc = 40.0
Ns = 60
n = 1.3
T = 298

q = 1.602e-19
k = 1.381e-23
Vt = (n * k * T) / q
Io = Isc / (np.exp(Voc / (Ns * Vt)) - 1)

# ==============================
# PARTIAL SHADING
# ==============================
G1 = 1000
G2 = 400

Vo = 80

def pv_current_module(V, G):
    Iph = Isc * (G / 1000)
    I = Iph - Io * (np.exp(V / (Ns * Vt)) - 1)
    return np.maximum(I, 0)

def pv_current_total(V):
    V1 = V / 2
    V2 = V / 2
    I1 = pv_current_module(V1, G1)
    I2 = pv_current_module(V2, G2)
    return np.minimum(I1, I2)

def fitness(D):
    Vpv = Vo * (1 - D)
    if Vpv <= 0 or Vpv >= 2*Voc:
        return 0
    return Vpv * pv_current_total(Vpv)

# ==============================
# PURE PSO
# ==============================
def run_pso():
    num_particles = 12
    max_iter = 25
    w, c1, c2 = 0.7, 1.5, 1.5

    D = np.random.uniform(0, 0.9, num_particles)
    velocity = np.zeros(num_particles)

    pbest = D.copy()
    pbest_val = np.array([fitness(d) for d in D])
    gbest = pbest[np.argmax(pbest_val)]

    history = []

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocity[i] = (w*velocity[i] 
                           + c1*r1*(pbest[i]-D[i]) 
                           + c2*r2*(gbest-D[i]))
            D[i] += velocity[i]
            D[i] = np.clip(D[i], 0, 0.9)

            P = fitness(D[i])
            if P > pbest_val[i]:
                pbest[i] = D[i]
                pbest_val[i] = P

        gbest = pbest[np.argmax(pbest_val)]
        history.append(max(pbest_val))

    return gbest, history

# ==============================
# PURE INC
# ==============================
def run_inc():
    D = 0.3
    step = 0.005
    history = []

    for _ in range(60):
        V1 = Vo*(1-D)
        I1 = pv_current_total(V1)
        P1 = V1*I1

        V2 = Vo*(1-(D+step))
        I2 = pv_current_total(V2)
        P2 = V2*I2

        dP = P2 - P1
        dV = V2 - V1

        history.append(P1)

        if abs(dP) < 1e-3:
            break

        if dP/dV > 0:
            D -= step
        else:
            D += step

        D = np.clip(D, 0, 0.9)

    return D, history

# ==============================
# HYBRID PSO + INC
# ==============================
def run_hybrid():
    D_pso, hist_pso = run_pso()

    D = D_pso
    step = 0.002
    history = hist_pso.copy()

    for _ in range(40):
        V1 = Vo*(1-D)
        I1 = pv_current_total(V1)
        P1 = V1*I1

        V2 = Vo*(1-(D+step))
        I2 = pv_current_total(V2)
        P2 = V2*I2

        dP = P2 - P1
        dV = V2 - V1

        history.append(P1)

        if abs(dP) < 1e-3:
            break

        if dP/dV > 0:
            D -= step
        else:
            D += step

        D = np.clip(D, 0, 0.9)

    return D, history

# ==============================
# RUN ALL METHODS
# ==============================
D_pso, hist_pso = run_pso()
D_inc, hist_inc = run_inc()
D_hybrid, hist_hybrid = run_hybrid()

P_pso = fitness(D_pso)
P_inc = fitness(D_inc)
P_hybrid = fitness(D_hybrid)

print("PURE PSO Power:", round(P_pso,2),"W")
print("PURE INC Power:", round(P_inc,2),"W")
print("HYBRID Power:", round(P_hybrid,2),"W")

# ==============================
# PLOT P-V CURVE
# ==============================
V = np.linspace(0,2*Voc,400)
P = V * pv_current_total(V)

plt.figure()
plt.plot(V,P)
plt.scatter(Vo*(1-D_pso),P_pso,label="PSO")
plt.scatter(Vo*(1-D_inc),P_inc,label="INC")
plt.scatter(Vo*(1-D_hybrid),P_hybrid,label="Hybrid")
plt.legend()
plt.xlabel("PV Voltage (V)")
plt.ylabel("Power (W)")
plt.title("MPPT Comparison Under Partial Shading")
plt.grid()
plt.show()

# ==============================
# CONVERGENCE PLOT
# ==============================
plt.figure()
plt.plot(hist_pso,label="PSO")
plt.plot(hist_inc,label="INC")
plt.plot(hist_hybrid,label="Hybrid")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Power (W)")
plt.title("Convergence Comparison")
plt.grid()
plt.show()