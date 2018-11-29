import numpy as np

from openmdao.api import ExplicitComponent

class HullFE(ExplicitComponent):

	def setup(self):
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('Z_l', val=np.zeros(10))

		self.add_output('f_Ea', val=np.zeros(10), units='MPa')
		self.add_output('f_Em', val=np.zeros(10), units='MPa')
		self.add_output('f_Eh', val=np.zeros(10), units='MPa')
		self.add_output('f_Etau', val=np.zeros(10), units='MPa')

		self.declare_partials('f_Ea', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Em', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Eh', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Etau', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Ea', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Em', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Eh', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Etau', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Ea', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Em', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Eh', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Etau', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Ea', 'Z_l', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Em', 'Z_l', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Eh', 'Z_l', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_Etau', 'Z_l', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		E = 2.1e5 #MPa
		nu = 0.3

		wt_spar = inputs['wt_spar']
		l_stiff = inputs['l_stiff']
		r_hull = inputs['r_hull']
		Z_l = inputs['Z_l']

		psi_Ea = 1.
		xi_Ea = 0.702 * Z_l
		rho_Ea = 0.5 * (1. + r_hull / (150. * wt_spar))**(-0.5)
		C_Ea = psi_Ea * np.sqrt(1. + (rho_Ea * xi_Ea / psi_Ea)**2.)
		f_Ea = C_Ea * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		psi_Em = 1.
		xi_Em = 0.702 * Z_l
		rho_Em = 0.5 * (1. + r_hull / (300. * wt_spar))**(-0.5)
		C_Em = psi_Em * np.sqrt(1. + (rho_Em * xi_Em / psi_Em)**2.)
		f_Em = C_Em * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		psi_Eh = 4.
		xi_Eh = 1.04 * np.sqrt(Z_l)
		rho_Eh = 0.6
		C_Eh = psi_Eh * np.sqrt(1. + (rho_Eh * xi_Eh / psi_Eh)**2.)
		f_Eh = C_Eh * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		psi_Etau = 5.34
		xi_Etau = 0.856 * Z_l**(3./4.)
		rho_Etau = 0.6
		C_Etau = psi_Etau * np.sqrt(1. + (rho_Etau * xi_Etau / psi_Etau)**2.)
		f_Etau = C_Etau * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		outputs['f_Ea'] = f_Ea
		outputs['f_Em'] = f_Em
		outputs['f_Eh'] = f_Eh
		outputs['f_Etau'] = f_Etau

	def compute_partials(self, inputs, partials):
		E = 2.1e5
		nu = 0.3

		wt_spar = inputs['wt_spar']
		l_stiff = inputs['l_stiff']
		r_hull = inputs['r_hull']
		Z_l = inputs['Z_l']

		psi_Ea = 1.
		xi_Ea = 0.702 * Z_l
		rho_Ea = 0.5 * (1. + r_hull / (150. * wt_spar))**(-0.5)
		C_Ea = psi_Ea * np.sqrt(1. + (rho_Ea * xi_Ea / psi_Ea)**2.)
		f_Ea = C_Ea * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		psi_Em = 1.
		xi_Em = 0.702 * Z_l
		rho_Em = 0.5 * (1. + r_hull / (300. * wt_spar))**(-0.5)
		C_Em = psi_Em * np.sqrt(1. + (rho_Em * xi_Em / psi_Em)**2.)
		f_Em = C_Em * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		psi_Eh = 4.
		xi_Eh = 1.04 * np.sqrt(Z_l)
		rho_Eh = 0.6
		C_Eh = psi_Eh * np.sqrt(1. + (rho_Eh * xi_Eh / psi_Eh)**2.)
		f_Eh = C_Eh * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		psi_Etau = 5.34
		xi_Etau = 0.856 * Z_l**(3./4.)
		rho_Etau = 0.6
		C_Etau = psi_Etau * np.sqrt(1. + (rho_Etau * xi_Etau / psi_Etau)**2.)
		f_Etau = C_Etau * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		partials['f_Ea', 'wt_spar'] = C_Ea * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * 1. / l_stiff + 0.25 * (1. + r_hull / (150. * wt_spar))**(-1.5) * r_hull / (150. * wt_spar**2.) * psi_Ea * 0.5 / np.sqrt(1. + (rho_Ea * xi_Ea / psi_Ea)**2.) * 2. * (rho_Ea * xi_Ea / psi_Ea) * xi_Ea / psi_Ea * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.
		partials['f_Ea', 'l_stiff'] = -C_Ea * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * wt_spar / l_stiff**2.
		partials['f_Ea', 'r_hull'] = -0.25 * (1. + r_hull / (150. * wt_spar))**(-1.5) * 1. / (150. * wt_spar) * psi_Ea * 0.5 / np.sqrt(1. + (rho_Ea * xi_Ea / psi_Ea)**2.) * 2. * (rho_Ea * xi_Ea / psi_Ea) * xi_Ea / psi_Ea * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.
		partials['f_Ea', 'Z_l'] = 0.702 * psi_Ea * 0.5 / np.sqrt(1. + (rho_Ea * xi_Ea / psi_Ea)**2.) * 2. * (rho_Ea * xi_Ea / psi_Ea) * rho_Ea / psi_Ea * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		partials['f_Em', 'wt_spar'] = C_Em * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * 1. / l_stiff + 0.25 * (1. + r_hull / (300. * wt_spar))**(-1.5) * r_hull / (300. * wt_spar**2.) * psi_Em * 0.5 / np.sqrt(1. + (rho_Em * xi_Em / psi_Em)**2.) * 2. * (rho_Em * xi_Em / psi_Em) * xi_Em / psi_Em * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.
		partials['f_Em', 'l_stiff'] = -C_Em * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * wt_spar / l_stiff**2.
		partials['f_Em', 'r_hull'] = -0.25 * (1. + r_hull / (300. * wt_spar))**(-1.5) * 1. / (300. * wt_spar) * psi_Em * 0.5 / np.sqrt(1. + (rho_Em * xi_Em / psi_Em)**2.) * 2. * (rho_Em * xi_Em / psi_Em) * xi_Em / psi_Em * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.
		partials['f_Em', 'Z_l'] = 0.702 * psi_Em * 0.5 / np.sqrt(1. + (rho_Em * xi_Em / psi_Em)**2.) * 2. * (rho_Em * xi_Em / psi_Em) * rho_Em / psi_Em * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		partials['f_Eh', 'wt_spar'] = C_Eh * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * 1. / l_stiff
		partials['f_Eh', 'l_stiff'] = -C_Eh * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * wt_spar / l_stiff**2.
		partials['f_Eh', 'r_hull'] = np.zeros(10)
		partials['f_Eh', 'Z_l'] = 1.04 * 0.5 / np.sqrt(Z_l) * psi_Eh * 0.5 / np.sqrt(1. + (rho_Eh * xi_Eh / psi_Eh)**2.) * 2. * (rho_Eh * xi_Eh / psi_Eh) * rho_Eh / psi_Eh * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.

		partials['f_Etau', 'wt_spar'] = C_Etau * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * 1. / l_stiff
		partials['f_Etau', 'l_stiff'] = -C_Etau * np.pi**2. * E / (12. * (1. - nu**2.)) * 2. * (wt_spar / l_stiff) * wt_spar / l_stiff**2.
		partials['f_Etau', 'r_hull'] = np.zeros(10)
		partials['f_Etau', 'Z_l'] = 0.856 * 3. / 4. * Z_l**(-1./4.) * psi_Etau * 0.5 / np.sqrt(1. + (rho_Etau * xi_Etau / psi_Etau)**2.) * 2. * (rho_Etau * xi_Etau / psi_Etau) * rho_Etau / psi_Etau * np.pi**2. * E / (12. * (1. - nu**2.)) * (wt_spar / l_stiff)**2.