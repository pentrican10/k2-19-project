# A Decade of Transit-Timing Measurements Confirm Resonance in the K2-19 System ([paper](https://arxiv.org/abs/2509.18031))
### Submitted to AJ
Abstract: K2-19 is a star, slightly smaller than the Sun, that hosts three transiting planets. Two of these, K2-19 b and c, are between the size of Neptune and Saturn and have orbital periods near a 3:2 commensurability, and exhibit strong transit-timing variations (TTVs). A previous TTV analysis reported moderate eccentricities of ≈0.20±0.03 for the two planets, but such high values would imply rapid orbital decay for the innermost planet d. Here, we present an updated analysis that includes eight new transit times from TESS, which extends the time baseline from three years to a decade, and employ a gradient-aware TTV modeling code ([jnkepler](https://github.com/kemasuda/jnkepler)). We confirm that the system resides in resonance with a small libration amplitude, but find a broader constraints on eccentricity that range from a few percent up to 0.2. These revised eccentricities alleviate previous concerns regarding rapid tidal circularization and support the long-term dynamical stability of the system.


- uses [jnkepler](https://github.com/kemasuda/jnkepler) v0.2.0
- Set up conda environmet using **environment.yml**
- run order:
    - 0_tess_limb_darkening.ipynb
    - 1_fit_transit_times.ipynb
    - 2_combine_data.ipynb
    - 3_jnkep_minimizer_fit.ipynb
    - 4_jnkep_fit_k2-19_nburn-1000_nsteps-1000_accept-90_tree-11_hessian_emax-0.5.py
    - 5_run_analysis.ipynb
    - 6_compare_models.ipynb
    - 7_resonant_figure.ipynb

- Complementary files:
    - Files starting with 'comp_'
    - Used for testing and setting up jnkepler mcmc run
 
