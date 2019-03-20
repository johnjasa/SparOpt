import numpy as np

from openmdao.api import Group

from prob_max_surge import ProbMaxSurge
from prob_max_pitch import ProbMaxPitch
from prob_max_tower_stress import ProbMaxTowerStress
from prob_max_fairlead import ProbMaxFairlead
from prob_max_moor_ten import ProbMaxMoorTen
from constr_50_surge import Constr50Surge
from constr_50_pitch import Constr50Pitch
from constr_50_tower_stress import Constr50TowerStress
from constr_50_fairlead import Constr50Fairlead
from constr_50_moor_ten import Constr50MoorTen

class ExtremeResponse(Group):

	def setup(self):
		
		self.add_subsystem('prob_max_surge', ProbMaxSurge(), promotes_inputs=['v_z_surge', 'mean_surge', 'stddev_surge'], promotes_outputs=['prob_max_surge'])

		self.add_subsystem('prob_max_pitch', ProbMaxPitch(), promotes_inputs=['v_z_pitch', 'mean_pitch', 'stddev_pitch'], promotes_outputs=['prob_max_pitch'])

		self.add_subsystem('prob_max_tower_stress', ProbMaxTowerStress(), promotes_inputs=['v_z_tower_stress', 'mean_tower_stress', 'stddev_tower_stress'], promotes_outputs=['prob_max_tower_stress'])

		self.add_subsystem('prob_max_fairlead', ProbMaxFairlead(), promotes_inputs=['v_z_fairlead', 'moor_offset', 'stddev_fairlead'], promotes_outputs=['prob_max_fairlead'])

		self.add_subsystem('prob_max_moor_ten', ProbMaxMoorTen(), promotes_inputs=['v_z_moor_ten', 'mean_moor_ten', 'stddev_moor_ten', 'gamma_F_moor_mean', 'gamma_F_moor_dyn'], promotes_outputs=['prob_max_moor_ten'])

		self.add_subsystem('constr_50_surge', Constr50Surge(), promotes_inputs=['prob_max_surge', 'maxval_surge'], promotes_outputs=['constr_50_surge'])

		self.add_subsystem('constr_50_pitch', Constr50Pitch(), promotes_inputs=['prob_max_pitch', 'maxval_pitch'], promotes_outputs=['constr_50_pitch'])

		self.add_subsystem('constr_50_tower_stress', Constr50TowerStress(), promotes_inputs=['prob_max_tower_stress', 'maxval_tower_stress'], promotes_outputs=['constr_50_tower_stress'])

		self.add_subsystem('constr_50_fairlead', Constr50Fairlead(), promotes_inputs=['prob_max_fairlead', 'maxval_fairlead'], promotes_outputs=['constr_50_fairlead'])

		self.add_subsystem('constr_50_moor_ten', Constr50MoorTen(), promotes_inputs=['prob_max_moor_ten', 'maxval_moor_ten'], promotes_outputs=['constr_50_moor_ten'])