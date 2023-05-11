# DCE-Signal-Intensity-Approximation-using-NODEs
## A Generative model for interpolating signals between observed DCE MRI phases in Breast CA

**Currently, I am at the very start of this project.** This model aimed to generated DCE-MRI signal at any timepoint(s).

## Initiative (Based on the ISPY trails)
### We are identifying some "features" or "characteristics" from the MRI scanning, and used to correlated with the patient's **Pathologic Complete Response (PCR)**. We currently have some great tools to "define" the so-called **Functional Tumor Volume (FTV)** from scanning, like PE, SER, Ktrans, AUC etc.'

In a nutshell, PE/SER defined FTV(s) does not really looks like a tumor volume and it discriminability in predicting PCR is much weaker than using Ktrans.

The measurement of Ktrans need a thing named **Arterial Input Function (AIF)**, which measures the tracer concentration in blood plasma at the first beginning. While this is not natively provided in ISPY-dataset. On top of that, the measurement of AIF put additional workload on **image acquisition** which much increase the complexitiy of the scanning and let alone its technical difficutly i.e. the old scanner may not able to perform it.

AUC (Area under the enhancement curve) is a good tool for identifying the "real" tumor volume from other tissues. However, usually a patient inside ISPY-dataset have 4-8 image phases acquired during the whole scanning (usually the time span is 600s). Obviously it is insufficient to plot an accurate Enhacement cureve with these data points.

--> Therefore, I am "trying" to make this AUC possible by generating pseudo-signals along its contrast enhanced time at a single voxel level, using the favor of NeuralODE.

~~A more realistic initiative is that I just wanna find something to improve my understanding on NeuralODE and it associated maths...... Forgive me...~~









