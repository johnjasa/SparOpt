import numpy as np

from openmdao.api import ExplicitComponent

class ECsExt(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		self.ECfile = EC['ECfile']

		self.add_output('windspeed_0_ext', val=np.zeros(self.N_EC), units='m/s')
		self.add_output('Hs_ext', val=np.zeros(self.N_EC), units='m')
		self.add_output('Tp_ext', val=np.zeros(self.N_EC), units='s')

	def compute(self,inputs,outputs):
		outputs['windspeed_0_ext'], outputs['Hs_ext'], outputs['Tp_ext'] = np.loadtxt(self.ECfile, unpack=True)