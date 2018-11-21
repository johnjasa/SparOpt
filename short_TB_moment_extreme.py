import numpy as np

from openmdao.api import ExplicitComponent

class ShortTBMomentExtreme(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		#self.add_input('mean_TB_moment', val=0.)
		self.add_input('stddev_TB_moment', val=0.)
		self.add_input('Nz_TB_moment', val=0.)

		self.add_output('extreme_TB_moment', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		#mu = inputs['mean_TB_moment']
		sigma = inputs['stddev_TB_moment']
		Nz = inputs['Nz_TB_moment']

		outputs['extreme_TB_moment'] = mu + sigma * np.sqrt(2. * np.log(Nz))
		

	def compute_partials(self, inputs, partials): #TODO check
		#mu = inputs['mean_TB_moment']
		sigma = inputs['stddev_TB_moment']
		Nz = inputs['Nz_TB_moment']

		outputs['extreme_TB_moment'] = mu + sigma * np.sqrt(2. * np.log(Nz))
		
		#partials['extreme_TB_moment', 'mean_TB_moment'] = 1.
		partials['extreme_TB_moment', 'stddev_TB_moment'] = np.sqrt(2. * np.log(Nz))
		partials['extreme_TB_moment', 'Nz_TB_moment'] = sigma * 1. / np.sqrt(2. * np.log(Nz)) * 1. / Nz
