import numpy as np

from openmdao.api import Group

from short_term_surge_cdf import ShortTermSurgeCDF
from short_term_pitch_cdf import ShortTermPitchCDF
from short_term_tower_stress_cdf import ShortTermTowerStressCDF
from short_term_My_shell_buckling_cdf import ShortTermMyShellBucklingCDF
from short_term_My_hoop_stress_cdf import ShortTermMyHoopStressCDF
from short_term_My_mom_inertia_cdf import ShortTermMyMomInertiaCDF

class ExtremeResponse(Group):

	def setup(self):
		
		self.add_subsystem('short_term_surge_cdf', ShortTermSurgeCDF(), promotes_inputs=['v_z_surge', 'mean_surge', 'stddev_surge', 'maxval_surge'], promotes_outputs=['short_term_surge_CDF'])

		self.add_subsystem('short_term_pitch_cdf', ShortTermPitchCDF(), promotes_inputs=['v_z_pitch', 'mean_pitch', 'stddev_pitch', 'maxval_pitch'], promotes_outputs=['short_term_pitch_CDF'])

		self.add_subsystem('short_term_tower_stress_cdf', ShortTermTowerStressCDF(), promotes_inputs=['v_z_tower_stress', 'mean_tower_stress', 'stddev_tower_stress', 'maxval_tower_stress'], promotes_outputs=['short_term_tower_stress_CDF'])

		self.add_subsystem('short_term_My_shell_buckling_cdf', ShortTermMyShellBucklingCDF(), promotes_inputs=['v_z_hull_moment', 'mean_hull_moment', 'stddev_hull_moment', 'maxval_My_shell_buckling'], promotes_outputs=['short_term_My_shell_buckling_CDF'])

		self.add_subsystem('short_term_My_hoop_stress_cdf', ShortTermMyHoopStressCDF(), promotes_inputs=['v_z_hull_moment', 'mean_hull_moment', 'stddev_hull_moment', 'maxval_My_hoop_stress'], promotes_outputs=['short_term_My_hoop_stress_CDF'])

		self.add_subsystem('short_term_My_mom_inertia_cdf', ShortTermMyMomInertiaCDF(), promotes_inputs=['v_z_hull_moment', 'mean_hull_moment', 'stddev_hull_moment', 'maxval_My_mom_inertia'], promotes_outputs=['short_term_My_mom_inertia_CDF'])