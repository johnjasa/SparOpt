import numpy as np

from openmdao.api import Group

from tower_diameter import TowerDiameter
from z_tower import ZTower

class Towerdim(Group):

	def setup(self):

		self.add_subsystem('tower_diameter', TowerDiameter(), promotes_inputs=['D_tower_p'], promotes_outputs=['D_tower'])

	 	self.add_subsystem('Z_tower', ZTower(), promotes_inputs=['L_tower'], promotes_outputs=['Z_tower'])