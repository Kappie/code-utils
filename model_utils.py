import hoomd
from hoomd import md





class Model(object):
    dim = 3
    folder_name = None

    def __init__(self, T=None, rho=None):
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

    def setup(self, types=None):
        if not types:
            types = self.particle_types

        # Neighbor list.
        nl = md.nlist.cell()

        # IPL interactions.
        epsilon = 1.0
        sigmaAA = 1.0; sigmaAB = 1.18; sigmaBB = 1.4;
        xcut = 1.48
        rcutAA = xcut * sigmaAA; rcutAB = xcut * sigmaAB; rcutBB = xcut * sigmaBB;

        ipl = md.pair.ipl_edan(nlist=nl, r_cut=rcutAA)
        ipl.pair_coeff.set(types[0], types[0], sigma=sigmaAA, epsilon=epsilon, r_cut=rcutAA)
        ipl.pair_coeff.set(types[0], types[1], sigma=sigmaAB, epsilon=epsilon, r_cut=rcutAB)
        ipl.pair_coeff.set(types[1], types[1], sigma=sigmaBB, epsilon=epsilon, r_cut=rcutBB)

        self.neighbor_list = nl


class IPL2D(IPL):
    name = '2dipl'
    folder_name = 'ipl'
    default_rho = 0.86
    dim = 2

    def setup(self, types=None):
        md.update.enforce2d()
        super().setup(types=types)



class StickySpheres(Model):
    particle_types = ['A', 'B']
    default_rho = 0.6
    name = 'sticky_spheres'
    dt = {1.2: 0.001, 0.95: 0.002, 0.80: 0.003, 0.70: 0.003, 0.60: 0.005}
    qmax = {0.95: [7.35, 6.4], 0.80: [6.86, 6.06], 0.70: [6.86, 5.87]}

    def setup(self, types=None):
        if not types:
            types = self.particle_types

        nl = md.nlist.cell()

        x_cut = 2**(1/6) * 1.2
        sigmaAA = 1.0; sigmaAB = 1.18; sigmaBB = 1.4;
        rcutAA = sigmaAA*x_cut; rcutAB = sigmaAB*x_cut; rcutBB = sigmaBB*x_cut;

        sticky = md.pair.sticky_spheres(nlist=nl, r_cut=rcutBB)
        sticky.pair_coeff.set(types[0], types[0], sigma=sigmaAA, r_cut=rcutAA)
        sticky.pair_coeff.set(types[0], types[1], sigma=sigmaAB, r_cut=rcutAB)
        sticky.pair_coeff.set(types[1], types[1], sigma=sigmaBB, r_cut=rcutBB)

        self.neighbor_list = nl

    def get_dt(self):
        return self.dt[self.rho]

    def get_qmax(self):
        return self.qmax[self.rho]


class StickySpheres2D(StickySpheres):
    name = '2d_sticky_spheres'
    folder_name = 'sticky_spheres'
    dim = 2

    dt = {0.67: 0.005}

    def setup(self, types=None):
        md.update.enforce2d()
        super().setup(types=types)


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


class ForceShiftedLJ(Model):
    name = "force_shifted_lj"
    folder_name = "force_shifted_lj"

    def setup(self):
        # Neighbor list.
        nl = md.nlist.cell()

        # modified KA interactions.
        epsilonAA = 1.0; sigmaAA = 1.0; r_cutAA = 1.5
        epsilonAB = 1.5; sigmaAB = 0.8; r_cutAB = 2.0
        epsilonBB = 0.5; sigmaBB = 0.88; r_cutBB = 1.5

        lj = md.pair.force_shifted_lj(nlist=nl, r_cut=r_cutAA)
        lj.pair_coeff.set('A', 'A', sigma=sigmaAA, epsilon=epsilonAA, r_cut=r_cutAA)
        lj.pair_coeff.set('A', 'B', sigma=sigmaAB, epsilon=epsilonAB, r_cut=r_cutAB)
        lj.pair_coeff.set('B', 'B', sigma=sigmaBB, epsilon=epsilonBB, r_cut=r_cutBB)

        self.neighbor_list = nl

    def get_dt(self):
        if self.T < 1.:
            return 0.005
        else:
            return 0.002

models = {
    'ipl': IPL,
    '2dipl': IPL2D,
    'sticky_spheres': StickySpheres,
    '2d_sticky_spheres': StickySpheres2D,
    'hertzian': Hertzian,
    'ipl_mono': IPLMono,
    '2dipl_mono': IPLMono2D,
    'force_shifted_lj': ForceShiftedLJ
}

