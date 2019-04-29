import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class BfbExt(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('dmoment_dv', val=0., units='N*s')
		self.add_input('x_d_towertop', val=0., units='m/m')
		self.add_input('Fdyn_tower_drag', val=0., units='N*s/m')
		self.add_input('Mdyn_tower_drag', val=0., units='N*s')

		self.add_output('Bfb_ext', val=np.ones((3,6)))

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		M_global = inputs['M_global']
		A_global = inputs['A_global']

		CoG_rotor = inputs['CoG_rotor']

		dthrust_dv = inputs['dthrust_dv'][0]
		dmoment_dv = inputs['dmoment_dv'][0]
		Fdyn_tower_drag = inputs['Fdyn_tower_drag'][0]
		Mdyn_tower_drag = inputs['Mdyn_tower_drag'][0]

		x_d_towertop = inputs['x_d_towertop']


		residuals['Bfb_ext'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Bfb_ext']) - np.array([[dthrust_dv + Fdyn_tower_drag, 0., 0., 1., 0., 0.],[CoG_rotor * dthrust_dv + Mdyn_tower_drag, dmoment_dv, 0., 0., 1., 0.],[dthrust_dv, x_d_towertop * dmoment_dv, 0., 0., 0., 1.]])

	def solve_nonlinear(self, inputs, outputs):
		M_global = inputs['M_global']
		A_global = inputs['A_global']

		CoG_rotor = inputs['CoG_rotor']

		dthrust_dv = inputs['dthrust_dv'][0]
		dmoment_dv = inputs['dmoment_dv'][0]
		Fdyn_tower_drag = inputs['Fdyn_tower_drag'][0]
		Mdyn_tower_drag = inputs['Mdyn_tower_drag'][0]

		x_d_towertop = inputs['x_d_towertop']

		outputs['Bfb_ext'] = np.matmul(np.linalg.inv(M_global + A_global),
			np.array([[dthrust_dv + Fdyn_tower_drag, 0., 0., 1., 0., 0.],
					  [CoG_rotor * dthrust_dv + Mdyn_tower_drag, dmoment_dv, 0., 0., 1., 0.],
					  [dthrust_dv, x_d_towertop * dmoment_dv, 0., 0., 0., 1.]], dtype='float'))

	def linearize(self, inputs, outputs, partials):
		CoG_rotor = inputs['CoG_rotor']

		dthrust_dv = inputs['dthrust_dv'][0]
		dmoment_dv = inputs['dmoment_dv'][0]
		Fdyn_tower_drag = inputs['Fdyn_tower_drag'][0]
		Mdyn_tower_drag = inputs['Mdyn_tower_drag'][0]

		x_d_towertop = inputs['x_d_towertop']

		partials['Bfb_ext', 'M_global'] = np.kron(np.identity(3),np.transpose(outputs['Bfb_ext']))
		partials['Bfb_ext', 'A_global'] = np.kron(np.identity(3),np.transpose(outputs['Bfb_ext']))
		partials['Bfb_ext', 'CoG_rotor'] = -np.array([np.zeros(6),[dthrust_dv, 0., 0., 0., 0., 0.],np.zeros(6)])
		partials['Bfb_ext', 'dthrust_dv'] = -np.array([[1., 0., 0., 0., 0., 0.],[CoG_rotor, 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0.]], dtype='float')
		partials['Bfb_ext', 'dmoment_dv'] = -np.array([np.zeros(6),[0., 1., 0., 0., 0., 0.],[0., x_d_towertop, 0., 0., 0., 0.]], dtype='float')
		partials['Bfb_ext', 'Fdyn_tower_drag'] = -np.array([[1., 0., 0., 0., 0., 0.],[0, 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0.]], dtype='float')
		partials['Bfb_ext', 'Mdyn_tower_drag'] = -np.array([np.zeros(6),[1., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0.]], dtype='float')
		partials['Bfb_ext', 'x_d_towertop'] = -np.array([np.zeros(6),np.zeros(6),[0., dmoment_dv, 0., 0., 0., 0.]], dtype='float')
		partials['Bfb_ext', 'Bfb_ext'] = np.kron(inputs['M_global'] + inputs['A_global'],np.identity(6))
