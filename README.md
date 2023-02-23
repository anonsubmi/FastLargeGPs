# FastOnlineGPsInLargeFields

Below is a list of all results in "Fast online Gaussian process inference for large geospatial fields", and which scripts will reproduce the results. Note that the time-results reported in "Fast online Gaussian process inference for large geospatial fields" are based on experiments on a Dell XPS 15 9560 laptop, with 16 GB RAM and an Intel Core i7-7700HQ CPU running at 2.80 GHz. The times may therefore be different if the code is run on a different computer. 

Figure 1: Illustrations/GenerateSurfaceIllustrations.m
Figure 2: Audio/CompareKLDivergences.m
Figure 3 + Table 1 times: 1st) Audio/VaryDomainSize.m, 2nd) Audio/PlotResultsVaryDomainSize.m
Figure 4: BathymetryData/runall.m
Figure 6 + Figure 7: Precipitation/EstimateDailyVaryDensities.m
Figure 8: 1st) Precipitation/EstimateDailyVaryDomainSizeAndTime.m, 2nd) Precipitation/PlotResultsEstimateDailyVaryDomainSize.m
Table 1 accuracies + MSLL: Audio/ComputeTableValues.m. (SMAE accuracy of SKI is based on reported value in [Yadav 2021]).
Table 2 accuracies: Precipitation/MCReps.m
Table 4: Precipitation/MCReps.m

[Yadav 2021]- Yadav, M., Sheldon, D., and Musco, C. Faster Kernel Interpolation for Gaussian Processes. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, pp. 2971–2979. PMLR, March 2021.




