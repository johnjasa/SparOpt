import numpy as np

from openmdao.api import ExplicitComponent

class AeroDamping(ExplicitComponent):

	def setup(self):
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('dmoment_dv', val=0., units='N*m*s/m')
		self.add_input('x_d_towertop', val=0., units='m/m')

		self.add_output('B_aero_11', val=0., units='N*s/m')
		self.add_output('B_aero_15', val=0., units='N*s')
		self.add_output('B_aero_17', val=0., units='N*s/m')
		self.add_output('B_aero_55', val=0., units='N*m*s')
		self.add_output('B_aero_57', val=0., units='N*s')
		self.add_output('B_aero_77', val=0., units='N*s/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		CoG_rotor = inputs['CoG_rotor']
		dthrust_dv = inputs['dthrust_dv']
		dmoment_dv = inputs['dmoment_dv']
		x_d_towertop = inputs['x_d_towertop']

		outputs['B_aero_11'] = dthrust_dv
		outputs['B_aero_15'] = CoG_rotor * dthrust_dv
		outputs['B_aero_17'] = dthrust_dv
		outputs['B_aero_55'] = CoG_rotor**2. * dthrust_dv + dmoment_dv
		outputs['B_aero_57'] = CoG_rotor * dthrust_dv + x_d_towertop * dmoment_dv
		outputs['B_aero_77'] = dthrust_dv + x_d_towertop**2. * dmoment_dv

	def compute_partials(self, inputs, partials):
		CoG_rotor = inputs['CoG_rotor']
		dthrust_dv = inputs['dthrust_dv']
		dmoment_dv = inputs['dmoment_dv']
		x_d_towertop = inputs['x_d_towertop']

		partials['B_aero_11', 'CoG_rotor'] = 0.
		partials['B_aero_11', 'dthrust_dv'] = 1.
		partials['B_aero_11', 'dmoment_dv'] = 0.
		partials['B_aero_11', 'x_d_towertop'] = 0.

		partials['B_aero_15', 'CoG_rotor'] = dthrust_dv
		partials['B_aero_15', 'dthrust_dv'] = CoG_rotor
		partials['B_aero_15', 'dmoment_dv'] = 0.
		partials['B_aero_15', 'x_d_towertop'] = 0.

		partials['B_aero_17', 'CoG_rotor'] = 0.
		partials['B_aero_17', 'dthrust_dv'] = 1.
		partials['B_aero_17', 'dmoment_dv'] = 0.
		partials['B_aero_17', 'x_d_towertop'] = 0.

		partials['B_aero_55', 'CoG_rotor'] = 2. * CoG_rotor * dthrust_dv
		partials['B_aero_55', 'dthrust_dv'] = CoG_rotor**2.
		partials['B_aero_55', 'dmoment_dv'] = 1.
		partials['B_aero_55', 'x_d_towertop'] = 0.

		partials['B_aero_57', 'CoG_rotor'] = dthrust_dv
		partials['B_aero_57', 'dthrust_dv'] = CoG_rotor
		partials['B_aero_57', 'dmoment_dv'] = x_d_towertop
		partials['B_aero_57', 'x_d_towertop'] = dmoment_dv

		partials['B_aero_77', 'CoG_rotor'] = 0.
		partials['B_aero_77', 'dthrust_dv'] = 1.
		partials['B_aero_77', 'dmoment_dv'] = x_d_towertop**2.
		partials['B_aero_77', 'x_d_towertop'] = 2. * x_d_towertop * dmoment_dv