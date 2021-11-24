from pyscf import gto,dft
import matplotlib.pyplot as plt
import numpy


# generate density in Gaussian cube format
# from pyscf.tools import cubegen
# dens = density(mol_hf, 'h2o_den.cube', mf_hf.make_rdm1()) #makes total density


def calc_density(mol, dm, nx=80, ny=80, nz=80):
    
    box_distance_from_atoms = 8

    coord = [mol.atom_coord(ia) for ia in range(mol.natm)]
    box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + box_distance_from_atoms
    boxorig = numpy.min(coord,axis=0) - box_distance_from_atoms/2
    
    xs = numpy.arange(nx) * (box[0]/nx)
    ys = numpy.arange(ny) * (box[1]/ny)
    zs = numpy.arange(nz) * (box[2]/nz)

    # physical box grid resulotion
    dx = box[0]/nx
    dy = (box[1]/ny)
    dz = (box[2]/nz)

    coords = numpy.vstack(numpy.meshgrid(xs,ys,zs)).reshape(3,-1).T
    coords = numpy.asarray(coords, order='C') - (-boxorig)
 
    nao = mol.nao_nr()
    ngrids = nx * ny * nz
    blksize = min(200, ngrids)
    rho = numpy.empty(ngrids)
    for ip0, ip1 in dft.gen_grid.prange(0, ngrids, blksize):
        ao = dft.numint.eval_ao(mol, coords[ip0:ip1])
        rho[ip0:ip1] = dft.numint.eval_rho(mol, ao, dm)
    
    total_vol = numpy.prod(box)
    tot_charge = total_vol*numpy.sum(rho)/len(rho)
    
    # calc dV element - h
    h = total_vol/len(rho)
    print("calc_density: total grid charge = {0} while number of electrons is {1}".format(tot_charge,numpy.sum(mol.nelec)))
    rho = rho.reshape(nx,ny,nz)
    return rho,h,(dx,dy,dz),box


def visualise_density(density, z_slice):

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    xx, yy = numpy.meshgrid(numpy.linspace(0, 1, 80), numpy.linspace(0, 1 , 80))

    X = xx
    Y = yy
    Z = 10 * numpy.ones(X.shape)

    dens_slice = dens[z_slice][:][:]

    data = dens_slice
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.imshow(data, cmap="plasma", interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
    plt.savefig('H2.png')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.contourf(X, Y, data, 100, zdir='z', offset=0.5, cmap="plasma")

    plt.show()

if __name__ == "__main__":

    mol_hf = gto.M(atom='Li 0 0 -0.544 ; H 0 0 0.544',basis = 'sto3g')
    mol_hf.verbose = 6

    mf_hf = dft.RKS(mol_hf)
    mf_hf.xc = 'lda,vwn' # default
    mf_hf = mf_hf.newton() # second-order algortihm
    s = mf_hf.kernel()

    dens,h,(dx,dy,dz),box2 = calc_density(mol_hf, mf_hf.make_rdm1()) #makes total density

    print("total charge is ",numpy.sum(dens) * h )

    visualise_density(dens,z_slice=39)