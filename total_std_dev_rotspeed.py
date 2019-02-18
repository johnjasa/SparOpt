import numpy as np

from openmdao.api import ExplicitComponent

class TotalStdDevRotspeed(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		
		for i in xrange(self.N_EC):
			self.add_input('stddev_rotspeed%d' % i, val=0.)
			self.add_input('p%d' % i, val=0.)

		self.add_output('total_stddev_rotspeed', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['total_stddev_rotspeed'] = 0.
		for i in xrange(self.N_EC):
			outputs['total_stddev_rotspeed'] += inputs['stddev_rotspeed%d' % i] * inputs['p%d' % i]
	
	def compute_partials(self, inputs, partials):
		for i in xrange(self.N_EC):
			partials['total_stddev_rotspeed', 'stddev_rotspeed%d' % i] = inputs['p%d' % i]
			partials['total_tower_fatigue_damage', 'p%d' % i] = inputs['stddev_rotspeed%d' % i]