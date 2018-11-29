import numpy as np

from openmdao.api import ExplicitComponent

class HullShearStress(ExplicitComponent):

	def setup(self):
		self.add_input('T_hull', val=np.zeros(10), units='N*m')
		self.add_input('Qy_hull', val=np.zeros(10), units='N')
		self.add_input('Qz_hull', val=np.zeros(10), units='N')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('angle_hull', val=0., units='rad')

		self.add_output('tau', val=np.zeros(10), units='MPa')

		self.declare_partials('tau', 'T_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('tau', 'Qy_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('tau', 'Qz_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('tau', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('tau', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('tau', 'angle_hull')

	def compute(self, inputs, outputs):
		T = inputs['T_hull']
		Qy = inputs['Qy_hull']
		Qz = inputs['Qz_hull']
		r_hull = inputs['r_hull']
		wt_spar = inputs['wt_spar']
		theta = inputs['angle_hull']

		tau_T = T / (2. * np.pi * r_hull**2. * wt_spar)
		tau_Q = -Qy / (np.pi * r_hull * wt_spar) * np.cos(theta) + Qz / (np.pi * r_hull * wt_spar) * np.sin(theta) 
		
		outputs['tau'] = abs(tau_T + tau_Q)

	def compute_partials(self, inputs, partials):
		T = inputs['T_hull']
		Qy = inputs['Qy_hull']
		Qz = inputs['Qz_hull']
		r_hull = inputs['r_hull']
		wt_spar = inputs['wt_spar']
		theta = inputs['angle_hull']

		partials['tau', 'T_hull'] = (tau_T + tau_Q) / abs(tau_T + tau_Q) * 1. / (2. * np.pi * r_hull**2. * wt_spar)
		partials['tau', 'Qy_hull'] = -(tau_T + tau_Q) / abs(tau_T + tau_Q) * 1. / (np.pi * r_hull * wt_spar) * np.cos(theta)
		partials['tau', 'Qz_hull'] = (tau_T + tau_Q) / abs(tau_T + tau_Q) * 1. / (np.pi * r_hull * wt_spar) * np.sin(theta)
		partials['tau', 'r_hull'] = (tau_T + tau_Q) / abs(tau_T + tau_Q) * (-2. * T / (2. * np.pi * r_hull**3. * wt_spar) + Qy / (np.pi * r_hull**2. * wt_spar) * np.cos(theta) - Qz / (np.pi * r_hull**2. * wt_spar) * np.sin(theta) )
		partials['tau', 'wt_spar'] = (tau_T + tau_Q) / abs(tau_T + tau_Q) * (-T / (2. * np.pi * r_hull**2. * wt_spar**2.) + Qy / (np.pi * r_hull * wt_spar**2.) * np.cos(theta) - Qz / (np.pi * r_hull * wt_spar**2.) * np.sin(theta) )
		partials['tau', 'angle_hull'] = (tau_T + tau_Q) / abs(tau_T + tau_Q) * (Qy / (np.pi * r_hull * wt_spar) * np.sin(theta) + Qz / (np.pi * r_hull * wt_spar) * np.cos(theta))