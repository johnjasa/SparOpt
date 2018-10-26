import numpy as np

from openmdao.api import ExplicitComponent

class GlobalDamping(ExplicitComponent):

	def setup(self):
		self.add_input('B_aero_11', val=0., units='N*s/m') #TODO: calculate these
		self.add_input('B_aero_15', val=0., units='N*s')
		self.add_input('B_aero_17', val=0., units='N*s/m')
		self.add_input('B_aero_55', val=0., units='N*m*s')
		self.add_input('B_aero_57', val=0., units='N*s')
		self.add_input('B_aero_77', val=0., units='N*s/m')
		self.add_input('B_struct_77', val=0., units='N*s/m')

		self.add_output('B_global', val=np.zeros((3,3)), units='N*s/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):

		outputs['B_global'] = np.zeros((3,3))

		outputs['B_global'][0,0] += inputs['B_aero_11']
		outputs['B_global'][0,1] += inputs['B_aero_15']
		outputs['B_global'][0,2] += inputs['B_aero_17']
		outputs['B_global'][1,0] += inputs['B_aero_15']
		outputs['B_global'][1,1] += inputs['B_aero_55']
		outputs['B_global'][1,2] += inputs['B_aero_57']
		outputs['B_global'][2,0] += inputs['B_aero_17']
		outputs['B_global'][2,1] += inputs['B_aero_57']
		outputs['B_global'][2,2] += inputs['B_aero_77'] + inputs['B_struct_77']

	def compute_partials(self, inputs, partials):
		partials['B_global', 'B_aero_11'] = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])
		partials['B_global', 'B_aero_15'] = np.array([0., 1., 0., 1., 0., 0., 0., 0., 0.])
		partials['B_global', 'B_aero_17'] = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0.])
		partials['B_global', 'B_aero_55'] = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
		partials['B_global', 'B_aero_57'] = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0.])
		partials['B_global', 'B_aero_77'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
		partials['B_global', 'B_struct_77'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])