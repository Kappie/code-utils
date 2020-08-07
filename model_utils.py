import hoomd
from hoomd import md





class Model(object):
    dim = 3
    folder_name = None

    def __init__(self, T, rho=None):
        self.T = T
        if rho == None:
            self.rho = self.default_rho
        else:
            self.rho = rho

        if self.folder_name is None:
            self.folder_name = self.name



class IPL(Model):
    particle_types = ['A', 'B']
    default_rho = 0.82
    name = 'ipl'

    def get_dt(self):
        if self.T < 1:
            return 0.005
        else:
            return 0.0025

    def setup(self):
        # Neighbor list.
        nl = md.nlist.cell()

        # IPL interactions.
        epsilon = 1.0
        sigmaAA = 1.0; sigmaAB = 1.18; sigmaBB = 1.4;
        xcut = 1.48
        rcutAA = xcut * sigmaAA; rcutAB = xcut * sigmaAB; rcutBB = xcut * sigmaBB;

        ipl = md.pair.ipl_edan(nlist=nl, r_cut=rcutAA)
        ipl.pair_coeff.set('A', 'A', sigma=sigmaAA, epsilon=epsilon, r_cut=rcutAA)
        ipl.pair_coeff.set('A', 'B', sigma=sigmaAB, epsilon=epsilon, r_cut=rcutAB)
        ipl.pair_coeff.set('B', 'B', sigma=sigmaBB, epsilon=epsilon, r_cut=rcutBB)

        self.neighbor_list = nl


class IPL2D(IPL):
    name = '2dipl'
    folder_name = 'ipl'
    default_rho = 0.86
    dim = 2

    def setup(self):
        md.update.enforce2d()
        super().setup()
        


class StickySpheres(Model):
    particle_types = ['A', 'B']
    default_rho = 0.6
    name = 'sticky_spheres'

    def setup(self):
        nl = md.nlist.cell()

        x_cut = 2**(1/6) * 1.2
        sigmaAA = 1.0; sigmaAB = 1.18; sigmaBB = 1.4;
        rcutAA = sigmaAA*x_cut; rcutAB = sigmaAB*x_cut; rcutBB = sigmaBB*x_cut;

        sticky = md.pair.sticky_spheres(nlist=nl, r_cut=rcutBB)
        sticky.pair_coeff.set('A', 'A', sigma=sigmaAA, r_cut=rcutAA)
        sticky.pair_coeff.set('A', 'B', sigma=sigmaAB, r_cut=rcutAB)
        sticky.pair_coeff.set('B', 'B', sigma=sigmaBB, r_cut=rcutBB)

        self.neighbor_list = nl

    def get_dt(self):
        if self.rho < 0.8:
            if self.T < 1:
                return 0.005
            else:
                return 0.0025
        else:
            return 0.001

class StickySpheres2D(StickySpheres):
    name = '2d_sticky_spheres'
    folder_name = 'sticky_spheres'
    dim = 2

    def setup(self):
        md.update.enforce2d()
        super().setup()


class Hertzian(Model):
    default_rho = 0.938
    name = 'hertzian'
    particle_types = ['A', 'B']

    def setup(self):
        nl = md.nlist.cell()

        rA = 0.5; rB = 0.7
        sigmaAA = 2*rA; sigmaAB = rA + rB; sigmaBB = 2*rB;
        rcutAA = sigmaAA; rcutAB = sigmaAB; rcutBB = sigmaBB;

        herzian = md.pair.hertzian(nlist=nl, r_cut=rcutAA)
        herzian.pair_coeff.set('A', 'A', sigma=sigmaAA, r_cut=rcutAA)
        herzian.pair_coeff.set('A', 'B', sigma=sigmaAB, r_cut=rcutAB)
        herzian.pair_coeff.set('B', 'B', sigma=sigmaBB, r_cut=rcutBB)

        self.neighbor_list = nl

    def get_dt(self):
        return 0.02



class IPLMono(Model):
    default_rho = 0.82
    name = 'ipl_mono'
    particle_types = ['A']

    def get_dt(self):
        if self.T < 1:
            return 0.005
        else:
            return 0.0025

    def setup(self):
        # Neighbor list.
        nl = md.nlist.cell()

        # IPL interactions.
        epsilon = 1.0
        sigmaAA = 1.0;
        xcut = 1.48
        rcutAA = xcut * sigmaAA;

        ipl = md.pair.ipl_edan(nlist=nl, r_cut=rcutAA)
        ipl.pair_coeff.set('A', 'A', sigma=sigmaAA, epsilon=epsilon, r_cut=rcutAA)

        self.neighbor_list = nl


class IPLMono2D(IPLMono):
    name = '2dipl_mono'
    folder_name = 'ipl_mono'
    default_rho = 0.86
    dim = 2

    def setup(self):
        md.update.enforce2d()
        super().setup()


models = {
    'ipl': IPL,
    '2dipl': IPL2D,
    'sticky_spheres': StickySpheres,
    '2d_sticky_spheres': StickySpheres2D,
    'hertzian': Hertzian,
    'ipl_mono': IPLMono,
    '2dipl_mono': IPLMono2D,
}

