import streamlit as st
import streamlit.components.v1 as components
import urllib.request
import numpy as np
import pandas as pd
import py3Dmol
import plotly.express as px
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import AllChem

# ==========================================
# 1. PAGE CONFIGURATION & UI (WITH FORM)
# ==========================================
st.set_page_config(page_title="GPCR Stable Dynamics", layout="wide", page_icon="🧬")

# Using st.form to prevent the app from rerunning on every slider move
with st.sidebar.form("sim_params"):
    st.header("⚙️ Simulation Settings")
    MC_STEPS = st.slider(
        "MC Micro-steps per Frame", min_value=50, max_value=500, value=150, step=50
    )
    TEMPERATURE = st.slider(
        "Temperature (kT)", min_value=0.1, max_value=10.0, value=2.0, step=0.5
    )

    st.markdown("---")
    POCKET_OPENING = st.slider(
        "Pocket Opening Amplitude", min_value=0.0, max_value=0.6, value=0.4, step=0.1
    )
    STEP_SIZE = st.slider(
        "Translation Step (Å)", min_value=0.1, max_value=2.0, value=0.5, step=0.1
    )
    ROT_SIZE = st.slider(
        "Rotation Step (Radians)", min_value=0.05, max_value=0.5, value=0.15, step=0.05
    )

    submit_btn = st.form_submit_button("🚀 Run Simulation", use_container_width=True)

st.title("🧬 Stable Dynamics: Sigmoidal Easing + Form UI")
st.markdown(
    """
**Update:** Added `st.form` for stable parameter control. 
Receptor deformation now utilizes a **Sigmoid** function, which prevents peptide bond breakage 
and graphic 'collapse' of the 3D model during the induced-fit process.
"""
)


# ==========================================
# 2. PHYSICS ENGINE (LJ, METROPOLIS, KINEMATICS)
# ==========================================
def safe_pdb_line(line, x, y, z):
    """Safely updates PDB ATOM coordinates maintaining strict Fortran column format."""
    if len(line) < 54:
        return line + "\n"
    return f"{line[:30]}{x:>8.3f}{y:>8.3f}{z:>8.3f}{line[54:].rstrip()}\n"


def format_hetatm(atom_id, sym, pos):
    """Formats HETATM lines for the ligand."""
    x, y, z = pos
    name_fmt = f" {sym:<3}"[:4]
    return f"HETATM{atom_id:>5} {name_fmt} ADN X   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {sym:>2}  \n"


def calculate_calibrated_energy(ligand_coords, receptor_coords):
    """Calculates Lennard-Jones (12-6) potential with an empirical desolvation penalty (Vina-like)."""
    if len(receptor_coords) == 0:
        return 0.0
    dists = np.clip(cdist(ligand_coords, receptor_coords), 1.2, 50.0)

    repulsion = (3.5 / dists) ** 12
    attraction = (3.5 / dists) ** 6
    raw_total_energy = np.sum(4 * 0.15 * (repulsion - attraction))

    # Apply Vina-like weight (0.45) only to favorable (negative) binding energy
    return raw_total_energy * 0.45 if raw_total_energy < 0 else raw_total_energy


def random_rotation_matrix(max_angle):
    """Generates a small random Euler rotation matrix for Monte Carlo steps."""
    angles = (np.random.rand(3) - 0.5) * 2 * max_angle
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    return rz @ ry @ rx


def monte_carlo_relaxation(
    lig_coords, rec_coords, anchor_point, temp, steps, trans_step, rot_step
):
    """Executes Monte Carlo simulated annealing to find local energy minima."""
    current_coords = lig_coords.copy()

    def total_energy(coords):
        lj_E = calculate_calibrated_energy(coords, rec_coords)
        com = np.mean(coords, axis=0)
        # Harmonic restraint pulling ligand towards the trajectory anchor
        return lj_E + 1.5 * np.sum((com - anchor_point) ** 2)

    current_total_E = total_energy(current_coords)
    best_coords = current_coords.copy()
    best_energy = current_total_E
    real_calibrated_energy = calculate_calibrated_energy(best_coords, rec_coords)

    accepted = 0
    for _ in range(steps):
        com = np.mean(current_coords, axis=0)
        # Apply random rotation and translation
        new_coords = (
            (current_coords - com) @ random_rotation_matrix(rot_step)
            + com
            + ((np.random.rand(3) - 0.5) * 2 * trans_step)
        )
        new_total_E = total_energy(new_coords)

        delta_E = new_total_E - current_total_E
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temp):
            current_coords = new_coords
            current_total_E = new_total_E
            accepted += 1
            if new_total_E < best_energy:
                best_energy = new_total_E
                best_coords = new_coords.copy()
                real_calibrated_energy = calculate_calibrated_energy(
                    best_coords, rec_coords
                )

    return best_coords, real_calibrated_energy, (accepted / max(1, steps)) * 100


# ==========================================
# 3. DATA PREPARATION (CACHED)
# ==========================================
@st.cache_data
def fetch_and_prepare_data():
    """Fetches receptor from RCSB PDB and generates Adenosine conformer via RDKit."""
    url = "https://files.rcsb.org/download/3EML.pdb"
    response = urllib.request.urlopen(url, timeout=10)
    receptor_lines = [
        line
        for line in response.read().decode("utf-8").split("\n")
        if line.startswith("ATOM")
    ]

    z_coords = [float(line[46:54]) for line in receptor_lines]
    pocket_center = np.array(
        [
            np.mean([float(line[30:38]) for line in receptor_lines]),
            np.mean([float(line[38:46]) for line in receptor_lines]),
            np.mean(z_coords),
        ]
    )

    # Generate Adenosine 3D structure
    mol = Chem.AddHs(
        Chem.MolFromSmiles(
            "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N"
        )
    )
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    # Center the ligand mathematically to prevent orbital rotation anomalies
    ligand_com = np.mean(
        [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], axis=0
    )
    ligand_template = [
        {
            "pos": np.array(
                [
                    conf.GetAtomPosition(a.GetIdx()).x,
                    conf.GetAtomPosition(a.GetIdx()).y,
                    conf.GetAtomPosition(a.GetIdx()).z,
                ]
            )
            - ligand_com,
            "sym": a.GetSymbol(),
        }
        for a in mol.GetAtoms()
    ]

    return (
        receptor_lines,
        pocket_center,
        (np.min(z_coords), np.max(z_coords)),
        ligand_template,
    )


# ==========================================
# 4. SIMULATION ENGINE (SIGMOID DEFORMATION)
# ==========================================
@st.cache_data
def run_full_simulation(mc_steps, temp, t_step, r_step, opening_force):
    """Runs the main trajectory loop combining Allostery and Monte Carlo docking."""
    receptor_lines, pocket_center, (min_z, max_z), ligand_template = (
        fetch_and_prepare_data()
    )
    cx, cy, cz = pocket_center

    frames = 50
    multi_model_pdb, metrics = "", []
    start_pos = pocket_center + np.array([25.0, 5.0, 20.0])
    end_pos = pocket_center + np.array([0.0, 0.0, 4.0])
    current_lig_coords = np.array([atom["pos"] + start_pos for atom in ligand_template])

    progress_bar = st.progress(0)

    for i in range(frames):
        progress = i / (frames - 1)
        anchor_point = start_pos + (end_pos - start_pos) * progress
        distance_to_pocket = np.linalg.norm(anchor_point - end_pos)

        receptor_shift = (
            (1.0 - (distance_to_pocket / 20.0)) * opening_force
            if distance_to_pocket < 20.0
            else 0.0
        )

        current_rec_coords = []
        current_frame_pdb = f"MODEL {i+1}\n"

        for line in receptor_lines:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            z_norm = (z - min_z) / (max_z - min_z + 1e-6)

            # THE MAGIC: Sigmoidal easing for smooth deformation!
            # Transitions from 1.0 (bottom) to 0.0 (top) preventing structural collapse.
            leverage = 1.0 / (1.0 + np.exp((z_norm - 0.4) * 15.0))

            new_x = x + ((x - cx) * leverage * receptor_shift)
            new_y = y + ((y - cy) * leverage * receptor_shift)

            angle = leverage * (receptor_shift * 0.1)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            fx = cx + (new_x - cx) * cos_a - (new_y - cy) * sin_a
            fy = cy + (new_x - cx) * sin_a + (new_y - cy) * cos_a

            current_rec_coords.append([fx, fy, z])
            current_frame_pdb += safe_pdb_line(line, fx, fy, z)

        current_rec_coords = np.array(current_rec_coords)
        # Optimize MC by calculating physics only against the local pocket (15 Å radius)
        local_receptor = current_rec_coords[
            np.linalg.norm(current_rec_coords - anchor_point, axis=1) < 15.0
        ]

        relaxed_coords, final_energy, acc_rate = monte_carlo_relaxation(
            current_lig_coords,
            local_receptor,
            anchor_point,
            temp,
            mc_steps,
            t_step,
            r_step,
        )
        current_lig_coords = relaxed_coords

        atom_id = 9000
        for idx, atom in enumerate(ligand_template):
            current_frame_pdb += format_hetatm(
                atom_id, atom["sym"], current_lig_coords[idx]
            )
            atom_id += 1

        current_frame_pdb += "ENDMDL\n"
        multi_model_pdb += current_frame_pdb

        metrics.append(
            {
                "Frame": i + 1,
                "Distance (Å)": np.linalg.norm(
                    np.mean(current_lig_coords, axis=0) - pocket_center
                ),
                "Binding Score (kcal/mol)": final_energy,
                "Acceptance Rate (%)": acc_rate,
            }
        )
        progress_bar.progress(int(progress * 100))

    progress_bar.empty()
    return multi_model_pdb, pd.DataFrame(metrics)


# ==========================================
# 5. UI RENDERING
# ==========================================
try:
    with st.spinner("Synthesizing conformations... (please wait for the progress bar)"):
        trajectory_pdb, df_metrics = run_full_simulation(
            MC_STEPS, TEMPERATURE, STEP_SIZE, ROT_SIZE, POCKET_OPENING
        )

    tab1, tab2 = st.tabs(["🖥️ 3D Simulation", "📈 Thermodynamics"])

    with tab1:
        view = py3Dmol.view(width=1000, height=600)
        view.addModelsAsFrames(trajectory_pdb)
        # Semi-transparent receptor to highlight the binding pocket
        view.setStyle({"model": -1}, {"cartoon": {"color": "spectrum", "opacity": 0.9}})
        view.setStyle(
            {"resn": "ADN"}, {"stick": {"colorscheme": "yellowCarbon", "radius": 0.2}}
        )

        view.animate({"loop": "forward", "interval": 120})
        view.zoomTo()
        components.html(view._make_html(), height=620, width=1000)

    with tab2:
        col_m1, col_m2 = st.columns(2)
        fig_ener = px.line(
            df_metrics,
            x="Frame",
            y="Binding Score (kcal/mol)",
            title="Calibrated Binding Energy (Vina-like ΔG)",
            markers=True,
            color_discrete_sequence=["#00FF7F"],
        )
        fig_ener.update_layout(template="plotly_dark", hovermode="x unified")
        col_m1.plotly_chart(fig_ener, use_container_width=True)

        fig_acc = px.bar(
            df_metrics,
            x="Frame",
            y="Acceptance Rate (%)",
            title="Accepted Micro-steps (Metropolis)",
            color_discrete_sequence=["#FFA500"],
        )
        fig_acc.update_layout(
            template="plotly_dark", hovermode="x unified", yaxis_range=[0, 100]
        )
        col_m2.plotly_chart(fig_acc, use_container_width=True)

except Exception as e:
    st.error(f"❌ Computation Error: {str(e)}")
