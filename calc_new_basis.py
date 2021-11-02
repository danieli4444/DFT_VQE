"""
construct ZM basis in order to create a density preserving basis to input to the VQE.


phi_i(r) = np.sqrt(dens/nelec) * np.exp(j * xsi_i(r))

xsi_i(r) = k_i * f(r) ;     k_i = 0 , +-1, +- 2 according to the number of wanted basis functions phi_i

f(r) = f(x,y,z) = 2pi/nelec * x_part + 2pi/(nelec *x_dens(x)) *y_part + 2pi/(nelec *xy_dens(x,y))*z_part ; notice f is a 3d function/array

x_part = scipy.integrate.sims(x_dens(x'))   - integrate  x' from 0 to x

y_part = scipy.integrate.sims(xy_dens(x,y'))  - integrate y' from 0 to y

z_part = scipy.integrate.sims(dens(x,y,z'))  - integrate z' from 0 to z


x_dens(x) = integrate density(x,y',z')
xy_dens(x,y) = integrate density(x,y, z')


Some simplifications:
1) For dx * dy * dz = h we assume dx=dy=dz , the grid spacings are equal



"""

import numpy as np
from pyscf import gto,dft

from pyscf_get_density import calc_density



def calc_x_dens(dens,dy,dz):
    """ compute density over x axis 

    Args:
        dens ([type]): [description]

    Returns:
        [type]: [description]
    """
    if dens.ndim != 3:
        raise("Wrong density array dimensions")
    x_dens = np.zeros(len(dens))
    for ix in range(len(dens)):
        x_dens[ix] = np.sum(dens[ix][:][:])
    x_dens *= dy*dz
    return x_dens


def calc_xy_dens(dens,dz):
    if dens.ndim != 3:
        raise("Wrong density array dimensions")
    xy_dens = np.zeros( (len(dens),len(dens[0])) )
    for ix in range(len(dens)):
        for iy in range(len(dens[0])):
            xy_dens[ix,iy] = np.sum(dens[ix][iy][:])
    xy_dens *= dz
    return xy_dens


def calc_x_part(x_dens,dens_shape,dx,nelec):
    """ returns a 3d np.array that contains the x_part of f(x,y,z) integral. notice that x_part is a 3d array

    Args:
        dens ([type]): 3d density
    """
    
    nx,ny,nz = dens_shape
    if len(dens_shape) != 3:
        raise("Wrong density array dimensions")

    x_part = np.zeros(dens_shape)

    for iy in range(ny):
        for iz in range(nz):
            x_part[:,iy,iz] = np.cumsum(x_dens[:]) * dx * (2*np.pi /nelec)

    return x_part

def calc_x_part2(x_dens,dens_shape,dx,nelec):
    pass

def calc_y_part(xy_density,x_dens,dens_shape,dy):
    """ returns a 3d np.array that contains the y_part of f(x,y,z) integral. notice that y_part is a 3d array

    Args:
        dens ([type]): [description]
    """
    if len(dens_shape) != 3:
        raise("Wrong density array dimensions")
    nx,ny,nz = dens_shape
    
    y_part = np.zeros(dens_shape)
    
    for ix in range(nx):
        for iz in range(nz):   
            y_part[ix,:,iz] = np.cumsum(xy_density[ix,:]) * dy  * (2*np.pi/x_dens[ix])

    return y_part



def calc_z_part(dens,xy_dens,dens_shape,dz):
    """ returns a 3d np.array that contains the y_part of f(x,y,z) integral. notice that y_part is a 3d array

    Args:
        dens ([type]): [description]
    """

    if len(dens_shape) != 3:
        raise("Wrong density array dimensions")  
    nx,ny,nz = dens_shape
    
    z_part = np.zeros(dens_shape)
  
    for ix in range(nx):   
        for iy in range(ny):
            z_part[ix,iy,:] = np.cumsum(dens[ix,iy,:]) * dz * (2*np.pi/xy_dens[ix,iy])

    return z_part


def check_basis(f,density,nelec,h):
    # k1 = 1
    # phi_1 = np.sqrt(density/nelec) * np.exp(1.j*k1* f)
    # # k2 = 3
    # phi_2 = np.sqrt(density/nelec) * np.exp(1.j*k2* f)
    k_ij = 3
    total_integral = 0
    for ix in range(len(density)):
        for iy in range(len(density[0])):
            for iz in range(len(density[0][0])):
                total_integral += density[ix,iy,iz] * np.exp(1.j * k_ij * f[ix,iy,iz])
    total_integral *= h/nelec
    print("check basis:")
    print("the total integral over <phi_i|phi_j> (kij = {0} ) = {1}".format(k_ij,total_integral))
 


if __name__ == "__main__":
    
  
    mol_hf = gto.M(atom='H 0 -0.544 0; H 0 0.544 0',basis = 'sto3g')
    mol_hf.verbose = 2

    mf_hf = dft.RKS(mol_hf)
    mf_hf.xc = 'lda,vwn' # default
    mf_hf = mf_hf.newton() # second-order algortihm
    s = mf_hf.kernel()

    #nelec = np.sum(test_density)

    axis_gridsize = 80
    nx = ny = nz = axis_gridsize

    # # 1. calc the 3d density
    dens,h,(dx,dy,dz) = calc_density(mol_hf, mf_hf.make_rdm1(),nx,ny,nz) #makes total density
    nelec = np.sum(dens) * h
    print("total charge is {0} and h = {1}".format(nelec,h))
    density_shape = dens.shape

  
    # # 2. calc the x_dens(x') and the xy_dens(x',y')
    x_dens = calc_x_dens(dens,dy,dz)
    #print(x_dens)
    import matplotlib.pyplot as plt
    plt.plot(x_dens)
    plt.show()
    xy_dens = calc_xy_dens(dens,dz)
  

    # 3. calc x,y,z parts cumulative integrals and assemble f(x,y,z)
    x_part = calc_x_part(x_dens,density_shape,dx,nelec) 
    y_part = calc_y_part(xy_dens,x_dens,density_shape,dy)
    z_part = calc_z_part(dens,xy_dens,density_shape,dz)

    f = x_part + y_part + z_part    

    check_basis(f,dens,nelec,h)
    

    print("finished")
