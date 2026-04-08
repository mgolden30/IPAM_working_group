'''
Simple test of low-fidelity methods to accelerate a high-fidelity
simulation.
'''

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

n = 128  #number of points
L = 22.0 #physical size of domain
steps = 1024*8

dt = 1.0
threshold = 1e-13



#Create a dimensionless coordinate x \in [0, 2*pi]
x = jnp.arange(n) / n * 2 * jnp.pi

#initial data
u = jnp.sin(x) + jnp.cos(2*x - 1) + 0.4*jnp.sin(3*x - 1.23)

def semi_implicit_step(u, dt):
    k = jnp.arange(n//2+1)
    mask = k < n/3
    k = k * 2 * jnp.pi / L
    
    uf = mask * jnp.fft.rfft(u)
    
    #advective term
    udu = mask * jnp.fft.rfft( jnp.fft.irfft(uf) * jnp.fft.irfft(uf * 1j * k))

    up = (uf - dt * udu) / (1 - dt*k**2 + dt*k**4) * mask

    return jnp.fft.irfft(up)


def implicit_euler_objective(up, u, dt):
    k = jnp.arange(n//2+1)
    mask = k < n/3
    k = k * 2 * jnp.pi / L
    
    #Return zero if we have a correct implicit Euler step
    upf = mask * jnp.fft.rfft(up)
    
    #advective term
    udu = mask * jnp.fft.rfft( jnp.fft.irfft(upf) * jnp.fft.irfft(upf * 1j * k))

    rhs = mask * jnp.fft.rfft(u) / dt - udu
    rhs = rhs / (1 - dt*k**2 + dt*k**4)

    f = upf/dt - rhs
    f = jnp.fft.irfft(f)
    return f

def implicit_step(u, dt, threshold, low_fidelity_model=None):
    #Use current state as initial guess
    if low_fidelity_model is None:
        up = u
    else:
        up = low_fidelity_model(u)

    F = lambda up: implicit_euler_objective(up, u, dt)

    def body_fn(carry):
        fevals, up = carry
        f = F(up)
        fevals = fevals + 1

        J = lambda v: jax.jvp(F, primals=(up,), tangents=(v,))[1]

        step, _ = jax.scipy.sparse.linalg.gmres(J, f, x0=None, tol=1e-5)

        #Take a full Newton step
        up = up - step
        return (fevals, up)

    def cond_fn(carry):
        fevals, up = carry
        f = F(up)
        return jnp.max(jnp.abs(f)) > threshold

    fevals, up = jax.lax.while_loop(cond_fn, body_fn, (0,up))
    return fevals, up

print("Running trivial model...")
def scan_fn(u, _):
    outputs = implicit_step(u, dt, threshold)
    up = outputs[1]
    return up, outputs
final, traj = jax.lax.scan(scan_fn, u, length=steps)

fevals = traj[0]
us = traj[1]
print(f"mean newton steps = {jnp.mean(fevals)}")
print("\n")


plt.imshow(us[:1024,:], aspect=0.1)
plt.title("u(x,t) KS")
plt.xlabel("x")
plt.ylabel("t")
plt.savefig("test.png")


plt.imshow(semi_implicit_step(us[:1024,:], dt), aspect=0.1)
plt.title("u(x,t) KS")
plt.xlabel("x")
plt.ylabel("t")
plt.savefig("si_test.png")


#Train a linear convolution
print("Learning a best fit linear model...")

us = jnp.fft.rfft(us)

x = us[:-1,:]
y = us[1: ,:]
kernel = jnp.sum(jnp.conj(x) * y, axis=0) / ( jnp.sum( jnp.conj(x) * x, axis=[0] ) + 1e-6)
norm = lambda x: jnp.linalg.norm( jnp.reshape(x, [-1,]))
y_pred = kernel * x
res = norm(y - y_pred) / norm(y)
print(f"linear model residual = {res}")

res = norm(y - x) / norm(y)
print(f"trival model ( up = u) residual = {res}")



#Apply the linear 
print("Testing linear model to generate initial guess...")
model = lambda u: jnp.fft.irfft( kernel * jnp.fft.rfft(u) )
def scan_fn(u, _):
    outputs = implicit_step(u, dt, threshold, low_fidelity_model=model)
    up = outputs[1]
    return up, outputs

final, traj = jax.lax.scan(scan_fn, u, length=steps)

fevals = traj[0]
print(f"mean newton steps = {jnp.mean(fevals)}")
print("\n")




print("Trying a linear model after a semi-implicit step")
us = jnp.fft.irfft(us)
us_si = semi_implicit_step( us[:-1,:], dt )
x = jnp.fft.rfft( us_si )
y = jnp.fft.rfft(us)[1: ,:]

kernel = jnp.sum(jnp.conj(x) * y, axis=0) / ( jnp.sum( jnp.conj(x) * x, axis=[0] ) + 1e-6)
norm = lambda x: jnp.linalg.norm( jnp.reshape(x, [-1,]))
y_pred = kernel * x
res = norm(y - y_pred) / norm(y)
print(f"semi-implicit residual = {res}")

model = lambda u: jnp.fft.irfft( kernel * jnp.fft.rfft( semi_implicit_step(u,dt)) )
def scan_fn(u, _):
    outputs = implicit_step(u, dt, threshold, low_fidelity_model=model)
    up = outputs[1]
    return up, outputs

final, traj = jax.lax.scan(scan_fn, u, length=steps)

fevals = traj[0]
us = traj[1]
print(f"mean newton steps = {jnp.mean(fevals)}")