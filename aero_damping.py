import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class AeroDamping(ExplicitComponent):

	def setup(self):
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('dmoment_dv', val=0., units='N*m*s/m')
		self.add_input('x_towermode', val=np.zeros(11), units='m')
		self.add_input('z_towermode', val=np.zeros(11), units='m')

		self.add_output('B_aero_11', val=0., units='N*s/m')
		self.add_output('B_aero_15', val=0., units='N*s')
		self.add_output('B_aero_17', val=0., units='N*s/m')
		self.add_output('B_aero_55', val=0., units='N*m*s')
		self.add_output('B_aero_57', val=0., units='N*s')
		self.add_output('B_aero_77', val=0., units='N*s/m')

	def compute(self, inputs, outputs):
		Z_tower = inputs['Z_tower']
		CoG_rotor = inputs['CoG_rotor']
		dthrust_dv = inputs['dthrust_dv']
		dmoment_dv = inputs['dmoment_dv']

		f_psi_tower = si.UnivariateSpline(inputs['z_towermode'], inputs['x_towermode'], s=0)
		f_psi_d_tower = f_psi_tower.derivative(n=1)

		outputs['B_aero_11'] = dthrust_dv + 169613.96114615502 #TODO
		outputs['B_aero_15'] = CoG_rotor * dthrust_dv
		outputs['B_aero_17'] = dthrust_dv
		outputs['B_aero_55'] = CoG_rotor**2. * dthrust_dv + dmoment_dv
		outputs['B_aero_57'] = CoG_rotor * dthrust_dv + f_psi_d_tower(Z_tower[-1]) * dmoment_dv
		outputs['B_aero_77'] = dthrust_dv + f_psi_d_tower(Z_tower[-1])**2. * dmoment_dv + 73852.60383701134 #TODO