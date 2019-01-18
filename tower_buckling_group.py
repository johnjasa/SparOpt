import numpy as np

from openmdao.api import Group

from tower_omega import TowerOmega
from tower_C_x import TowerCx
from tower_crit_buckling_stress import TowerCritBucklingStress
from tower_lambda_x import TowerLambdaX
from tower_alpha_x import TowerAlphaX
from tower_chi_x import TowerChiX
from tower_buckling_resist import TowerBucklingResist

class TowerBuckling(Group):

	def setup(self):
		
		self.add_subsystem('tower_omega', TowerOmega(), promotes_inputs=['L_tower', 'D_tower_p', 'wt_tower_p'], promotes_outputs=['tower_omega'])

		self.add_subsystem('tower_C_x', TowerCx(), promotes_inputs=['tower_omega', 'D_tower_p', 'wt_tower_p'], promotes_outputs=['C_x'])

		self.add_subsystem('tower_crit_buckling_stress', TowerCritBucklingStress(), promotes_inputs=['D_tower_p', 'wt_tower_p', 'C_x'], promotes_outputs=['sigma_x_Rcr'])

		self.add_subsystem('tower_lambda_x', TowerLambdaX(), promotes_inputs=['f_y', 'sigma_x_Rcr'], promotes_outputs=['lambda_x'])

		self.add_subsystem('tower_alpha_x', TowerAlphaX(), promotes_inputs=['D_tower_p', 'wt_tower_p'], promotes_outputs=['alpha_x'])

		self.add_subsystem('tower_chi_x', TowerChiX(), promotes_inputs=['lambda_x', 'alpha_x'], promotes_outputs=['chi_x'])

		self.add_subsystem('tower_buckling_resist', TowerBucklingResist(), promotes_inputs=['chi_x', 'f_y', 'gamma_M_tower', 'gamma_F_tower'], promotes_outputs=['maxval_tower_stress'])