# Evaluating Planet Vetting Robustness Across TESS Photometric Pipelines

How do automated planet-vetting tools perform when applied to light curves produced by different TESS photometric pipelines? Using a curated sample of known transiting exoplanets observed by TESS, I retrieve light curves from multiple pipelines (e.g., SPOC, QLP, TGLC, Eleanor, NEMESIS), standardize their formats, and apply a consistent set of quality control and vetting algorithms to each dataset. The primary goal is to quantify how pipeline-dependent choices, such as detrending methods, apertures, and noise characteristics, affect common exoplanet transit vetting metrics (e.g., signal-to-noise, detection efficiency, and planet/false-positive classification). I compare recovery rates and diagnostic metrics across pipelines, identify cases of disagreement, and investigate the underlying causes (e.g., systematics, crowding, or data gaps). This project emphasizes reproducible data analysis, statistical comparison of methods, and critical evaluation of automated tools commonly used in exoplanet discovery pipelines.


Link to Dax's Google Drive (where data is stored as a .zip file. Zip file ~24 GB but uncompressed, the data is ~ 40 GB!!):
[https://drive.google.com/drive/folders/1dE2YAE-RhTm0Ip1IXbZDaeAFX6Vmw2HA?usp=sharing](https://drive.google.com/drive/folders/1dE2YAE-RhTm0Ip1IXbZDaeAFX6Vmw2HA?usp=sharing)

TICs_with_no_LCs = [22233480, 39926974, 261257684, 336961891, 203214081, 19028197, 
                    150096001, 434226736, 165202476, 143032776, 324609476, 203289099,
                    173103335, 95337971, 437054764, 4619242, 33397739, 126982221, 16005254,
                    26054627, 39143128, 56798909, 458686847, 330637910, 388804061, 323687123,
                    252481136, 459762279, 305506996, 258037656, 52005579, 288144647, 85281192,
                    16550540, 443823169, 262470965, 291013124, 28872266, 63100069, 337217173,
                    57147191 446166017, 22740615 282113152]

TICs_with_T0_drift = [192833836, 111778581, 125405602, 243185500, 155867025, 230129753, 
                      126606859, 335590096, 328513434, 168936945, 230982415, 67512645,
                      77490011]
