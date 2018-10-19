import numpy as np
from scipy import linalg
import scipy.interpolate as si

from openmdao.api import ImplicitComponent

class BfbExt(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('dmoment_dv', val=0., units='N*s')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('x_towermode', val=np.zeros(11), units='m')
		self.add_input('z_towermode', val=np.zeros(11), units='m')

		self.add_output('Bfb_ext', val=np.zeros((3,6)))

	def apply_nonlinear(self, inputs, outputs, residuals):
		M_global = inputs['M_global']
		A_global = inputs['A_global']

		CoG_rotor = inputs['CoG_rotor']

		dthrust_dv = inputs['dthrust_dv'][0]
		dmoment_dv = inputs['dmoment_dv'][0]

		Z_tower = inputs['Z_tower']

		f_psi_tower = si.UnivariateSpline(inputs['z_towermode'], inputs['x_towermode'], s=0)
		f_psi_d_tower = f_psi_tower.derivative(n=1)

		residuals['Bfb_ext'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Bfb_ext']) - np.array([[dthrust_dv, 0., 0., 1., 0., 0.],[CoG_rotor * dthrust_dv, dmoment_dv, 0., 0., 1., 0.],[dthrust_dv, f_psi_d_tower(Z_tower[-1]) * dmoment_dv, 0., 0., 0., 1.]])

	def solve_nonlinear(self, inputs, outputs):
		M_global = inputs['M_global']
		A_global = inputs['A_global']

		CoG_rotor = inputs['CoG_rotor']

		dthrust_dv = inputs['dthrust_dv'][0]
		dmoment_dv = inputs['dmoment_dv'][0]

		Z_tower = inputs['Z_tower']

		f_psi_tower = si.UnivariateSpline(inputs['z_towermode'], inputs['x_towermode'], s=0)
		f_psi_d_tower = f_psi_tower.derivative(n=1)

		outputs['Bfb_ext'] = np.matmul(np.linalg.inv(M_global + A_global), np.array([[dthrust_dv, 0., 0., 1., 0., 0.],[CoG_rotor * dthrust_dv, dmoment_dv, 0., 0., 1., 0.],[dthrust_dv, f_psi_d_tower(Z_tower[-1]) * dmoment_dv, 0., 0., 0., 1.]]))