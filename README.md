#  GPCR Dynamic Docking & Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![RDKit](https://img.shields.io/badge/RDKit-Chemoinformatics-green)
![SciPy](https://img.shields.io/badge/SciPy-Physics_Engine-lightgrey)

An advanced, interactive web application built with Streamlit that simulates and analyzes the dynamic binding process of a ligand (Adenosine) to a G-Protein-Coupled Receptor (GPCR - ADORA2A). 

Unlike static rigid-body viewers, this dashboard implements real-time stochastic optimization and conformational receptor shifts to model the **Induced-Fit** mechanism.

##  Simulation Showcase

**1. The Approach (Pre-Docking)**
Before the ligand enters the binding site, the receptor's pocket begins to open via sigmoidal conformational easing.
![Pre-Docking Phase]<img width="1844" height="874" alt="image" src="https://github.com/user-attachments/assets/92cccf6f-013f-4a48-ad79-6dcade0d3cf2" />


**2. The Induced-Fit (Post-Docking)**
The Monte Carlo engine finds the optimal local energy minimum, safely navigating steric clashes within the newly expanded pocket.
![Post-Docking Phase]<img width="1852" height="856" alt="image" src="https://github.com/user-attachments/assets/4210a878-cb19-40b1-a0db-633793feb1b8" />

**3. Real-Time Thermodynamics Analytics**
Tracking the calibrated Lennard-Jones potential (Vina-like $\Delta G$) and Metropolis acceptance rate across the simulation timeline.
![Analytics Dashboard]<img width="1852" height="856" alt="image" src="https://github.com/user-attachments/assets/4210a878-cb19-40b1-a0db-633793feb1b8" />


## ✨ Key Features

* **Stochastic Ligand Optimization (Monte Carlo):** Uses Euler rotation matrices and translational shifts coupled with the **Metropolis criterion** to find local energy minima within the binding pocket.
* **Sigmoidal Conformational Easing:** Simulates the allosteric opening of the receptor. Uses a sigmoidal function to smoothly propagate coordinate shifts, preventing peptide bond breakage and 3D rendering artifacts.
* **Empirical Thermodynamic Scoring:** Calculates real-time Van der Waals interactions using the **Lennard-Jones (12-6) potential** via `scipy.spatial`. The raw energy is calibrated with an empirical weight to simulate desolvation penalties, mimicking **AutoDock Vina** scoring behavior.
* **Interactive 3D Visualization:** Seamlessly integrates `py3Dmol` for smooth, real-time rendering of the dynamic PDB trajectory.

## 🛠️ Technology Stack

* **Frontend/UI:** Streamlit, Streamlit-Components
* **Biophysics Engine:** NumPy, SciPy
* **Chemoinformatics:** RDKit
* **Visualization:** py3Dmol, Plotly
* **Data Handling:** Pandas, Urllib

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Bodya-collab/GPCR-Dynamic-Docking.git](https://github.com/Bodya-collab/GPCR-Dynamic-Docking.git)
   cd GPCR-Dynamic-Docking
