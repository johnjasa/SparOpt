import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class AeroDamping(ExplicitComponent):

	def setup(self):
		self.add_input('psi_d_top', val=0., units='m/m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('dmoment_dv', val=0., units='Nm*s/m')
		self.add_input('x_towermode', val=0., units='m')
		self.add_input('z_towermode', val=0., units='m')

		self.add_output('B_aero_11', val=0., units='N*s/m')
		self.add_output('B_aero_15', val=0., units='N*s/m')
		self.add_output('B_aero_17', val=0., units='N*s/m')
		self.add_output('B_aero_55', val=0., units='N*s/m')
		self.add_output('B_aero_57', val=0., units='N*s/m')
		self.add_output('B_aero_77', val=0., units='N*s/m')

	def compute(self, inputs, outputs):
		psi_d_top = inputs['psi_d_top']
		Z_tower = inputs['Z_tower']
		CoG_rotor = inputs['CoG_rotor']
		dthrust_dv = inputs['dthrust_dv']
		dmoment_dv = inputs['dmoment_dv']

		f_psi_tower = si.UnivariateSpline(inputs['z_towermode'], inputs['x_towermode'], s=0)
		f_psi_d_tower = f_psi_tower.derivative(n=1)

		outputs['B_aero_1'] = dthrust_dv
		outputs['B_aero_1'] = CoG_rotor * dthrust_dv
		outputs['B_aero_1'] = dthrust_dv
		outputs['B_aero_5'] = CoG_rotor**2. * dthrust_dv + dmoment_dv
		outputs['B_aero_5'] = CoG_rotor * dthrust_dv + f_psi_d_tower(Z_tower[-1]) * dmoment_dv
		outputs['B_aero_7'] = dthrust_dv + f_psi_d_tower(Z_tower[-1])**2. * dmoment_dv