import hoomd
from hoomd import md





class Model(object):
    def __init__(self, T, rho=None):
        self.T = T
        if rho == None:
            self.rho = self.default_rho
        else:
            self.rho = rho


class IPL(Model):
    default_rho = 0.82

    def get_dt(self):
        if self.T < 1:
            return 0.005
        else:
            return 0.0025

    def setup(self):
        # Neighbor list.
        optimal_r_buff = 0.15   # workstation GPU, T = 0.53, N = 2000.
        nl = md.nlist.cell(r_buff=optimal_r_buff)

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

class StickySpheres(Model):
    default_rho = 0.6

    def setup(self):
        optimal_r_buff = 0.17   # workstation GPU, T = 0.77, N = 3000.
        nl = md.nlist.cell(r_buff=optimal_r_buff)

        x_cut = 2**(1/6) * 1.2
        sigmaAA = 1.0; sigmaAB = 1.18; sigmaBB = 1.4;
        rcutAA = sigmaAA*x_cut; rcutAB = sigmaAB*x_cut; rcutBB = sigmaBB*x_cut;

        sticky = md.pair.sticky_spheres(nlist=nl, r_cut=rcutBB)
        sticky.pair_coeff.set('A', 'A', sigma=sigmaAA, r_cut=rcutAA)
        sticky.pair_coeff.set('A', 'B', sigma=sigmaAB, r_cut=rcutAB)
        sticky.pair_coeff.set('B', 'B', sigma=sigmaBB, r_cut=rcutBB)

        self.neighbor_list = nl

    def get_dt(self):
        if self.T < 1:
            return 0.005
        else:
            return 0.0025


class Hertzian(Model):
    default_rho = 0.938

    def setup(self):
        optimal_r_buff = 0.34 # workstation GPU, T = 0.0018, N = 4000
        nl = md.nlist.cell(r_buff=optimal_r_buff)

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




models = {
    'ipl': IPL,
    'sticky_spheres': StickySpheres,
    'hertzian': Hertzian
}
