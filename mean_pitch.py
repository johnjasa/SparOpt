import numpy as np

from openmdao.api import ExplicitComponent

class MeanPitch(ExplicitComponent):

	def setup(self):
		self.add_input('thrust_0', val=0., units='N')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('CoB', val=0., units='m')
		self.add_input('mass_total', val=0., units='kg')
		self.add_input('CoG_total', val=0., units='m')

		self.add_output('mean_pitch', val=0., units='rad')

	def compute(self, inputs, outputs):
		thrust_0 = inputs['thrust_0']
		CoG_rotor = inputs['CoG_rotor']
		z_moor = inputs['z_moor']
		buoy_spar = inputs['buoy_spar']
		CoB = inputs['CoB']
		mass_total = inputs['mass_total']
		CoG_total = inputs['CoG_total']

		outputs['mean_pitch'] = np.arcsin(thrust_0 * (CoG_rotor - z_moor) / (buoy_spar * (CoB - z_moor) - mass_total * 9.80665 * (CoG_total - 77.2)))