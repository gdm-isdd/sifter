#python_interactions_library

import os
import requests
import numpy as np
from collections import defaultdict
from Bio.PDB import PDBParser, NeighborSearch, Selection
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import pandas as pd
from rdkit.Chem import rdmolfiles
import pymol
from pymol import cmd
import webbrowser


################################    FUNCTION FOR LOADING THE DATASET OR COMPLEX ##############################################################################################

# Function to load PDBbind from the refined-set
def load_pdbbind_refined_data(data_dir):
    return [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


# Function that fetch resolution for a given PDB ID
def fetch_resolution(pdb_id, cache):
    if pdb_id in cache:
        return cache[pdb_id]  # Uses cached resolution if available
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        resolution = data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0]
        cache[pdb_id] = resolution  # Caches the resolution
        return resolution
    else:
        print(f"Error fetching data for {pdb_id}: {response.status_code}")
        cache[pdb_id] = None  # Cache the possible failed lookup as None
        return None


# Function to fetch resolutions for all PDB IDs concurrently (iterates fetch_resolution)
def fetch_all_resolutions(pdb_ids):
    resolution_cache = {} #dict for storing the resolutions
    with ThreadPoolExecutor(max_workers=10) as executor:  # Parallizes the fetching, the number of workers is the number of threads
        results = list(executor.map(lambda pdb_id: (pdb_id, fetch_resolution(pdb_id, resolution_cache)), pdb_ids)) #executor.map exploits the threads, lambda is a function that wraps resolutions to complexes (list of tuples)
    return dict(results)  # Convert list of tuples to a dictionary


# Function to load ligands
def load_ligand(complex_dir):
    ligand_file = None
    for file_name in os.listdir(complex_dir):
        if file_name.endswith('.mol') or file_name.endswith('.sdf'):
            ligand_file = os.path.join(complex_dir, file_name)
            print(f"Found ligand file: {file_name}")
            break

    if ligand_file is None:
        print(f"No ligand file found in {complex_dir}")
        return None

    if ligand_file.endswith('.mol'):
        ligand_mol = Chem.MolFromMolFile(ligand_file, removeHs=False, sanitize=False)
    elif ligand_file.endswith('.sdf'):
        ligand_supplier = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=False)
        ligand_mol = ligand_supplier[0] if ligand_supplier else None
    elif ligand_file.endswith('.mol2'):
        ligand_mol = Chem.MolFromMol2File(ligand_file, removeHs=False, sanitize=False)
    else:
        print(f"Unsupported file format for {ligand_file}")
        return None

    if ligand_mol is None:
        print(f"Failed to load ligand from {ligand_file}")
        return None

    # Attempt to sanitize and kekulize
    try:
        Chem.SanitizeMol(ligand_mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE) # avoids sanitization
        Chem.Kekulize(ligand_mol, clearAromaticFlags=True)
        ligand_mol = Chem.AddHs(ligand_mol)
        AllChem.ComputeGasteigerCharges(ligand_mol)
    except Exception as e:
        print(f"Error processing ligand in {complex_dir}: {e}")
        return None

    return ligand_mol


# Function to load proteins
def load_protein(complex_dir):
    pocket_pdb_file = None
    protein_pdb_file = None
    for file_name in os.listdir(complex_dir):
        if file_name.endswith('_pocket.pdb'):
            pocket_pdb_file = os.path.join(complex_dir, file_name)
        elif file_name.endswith('.pdb'):
            protein_pdb_file = os.path.join(complex_dir, file_name)

    pdb_file = pocket_pdb_file if pocket_pdb_file else protein_pdb_file
    if pdb_file is None:
        print(f"No protein PDB file found in {complex_dir}")
        return None

    parser = PDBParser(QUIET=True) #False if I want to see the warnings
    try:
        structure = parser.get_structure('', pdb_file)
    except Exception as e:
        print(f"Error loading protein structure in {complex_dir}: {e}")
        return None

    # Remove water molecules
    for model in structure:
        for chain in model:
            residues_to_remove = []
            for residue in chain:
                if residue.get_resname() == 'HOH':
                    residues_to_remove.append(residue)
            for residue in residues_to_remove:
                chain.detach_child(residue.id)

    return structure


# Function to prepare structures
def prepare_structures(complex_dirs, resolution_cache, res_treshold):
    prepared_structures = []
    skipped_folders = []
    skipped_folders_res = []

    for complex_dir in complex_dirs:
        pdb_id = os.path.basename(complex_dir)
        try:
            print(f"Processing folder: {pdb_id}")

            resolution = resolution_cache.get(pdb_id)
            if resolution is None or resolution > res_treshold:
                print(f"Skipping folder {pdb_id} due to resolution ({resolution} Å)")
                skipped_folders_res.append(pdb_id)
                continue

            ligand_mol = load_ligand(complex_dir)
            if ligand_mol is None:
                print(f"Ligand loading failed for folder: {pdb_id}")
                skipped_folders.append(pdb_id)
                continue

            protein_structure = load_protein(complex_dir)
            if protein_structure is None:
                print(f"Protein loading failed for folder: {pdb_id}")
                skipped_folders.append(pdb_id)
                continue

            prepared_structures.append({
                'protein': protein_structure,
                'ligand': ligand_mol,
                'id': pdb_id,
                'resolution': resolution
            })

        except Exception as e:
            print(f"Error processing folder {pdb_id}: {e}")
            skipped_folders.append(pdb_id)

    return prepared_structures, skipped_folders, skipped_folders_res


################################    FUNCTION FOR VISUALIZING A SINGLE LOADED COMPLEX ##############################################################################################


import nglview as nv
from Bio.PDB import PDBIO
from rdkit.Chem import MolToPDBBlock

# Function to save the protein as a PDB file
def save_protein_to_pdb(protein_structure, output_file="protein.pdb"):
    io = PDBIO()
    io.set_structure(protein_structure)
    io.save(output_file)
    return output_file

# Function to save the ligand as a PDB file
def save_ligand_to_pdb(ligand_mol, output_file="ligand.pdb"):
    pdb_block = MolToPDBBlock(ligand_mol)
    with open(output_file, 'w') as f:
        f.write(pdb_block)
    return output_file



################################    FUNCTION FOR SIFt and atom properties   ##############################################################################################


def determine_sift_interactions(prepared_structure, cutoff):
    protein_structure = prepared_structure['protein']
    ligand_mol = prepared_structure['ligand']

    # Gets the ligand atoms and coordinates
    ligand_conformer = ligand_mol.GetConformer()
    ligand_coords = np.array([ligand_conformer.GetAtomPosition(i)
                              for i in range(ligand_mol.GetNumAtoms())])

    # Gets protein atoms and coordinates
    protein_atoms = Selection.unfold_entities(protein_structure, 'A')  # Get all atoms

    # Creates NeighborSearch object for efficient distance calculation
    ns = NeighborSearch(protein_atoms)

    # Assigns properties to ligand atoms
    ligand_atom_properties = {}
    for i, ligand_atom in enumerate(ligand_mol.GetAtoms()):
        properties = {
            'is_hydrogen': ligand_atom.GetSymbol() == 'H', ####
            'is_donor': is_hbond_donor(ligand_atom),
            'is_acceptor': is_hbond_acceptor(ligand_atom),
            'is_aromatic': is_aromatic_atom(ligand_atom),
            'is_hydrophobic': is_hydrophobic_atom(ligand_atom),
            'is_positive': is_positive_atom(ligand_atom),
            'is_negative': is_negative_atom(ligand_atom),
        }
        print(properties)
        ligand_atom_properties[i] = properties
    print("ACTUAL REPORTED PROPERTIES\n\n")
    print(ligand_atom_properties)
    

    # Assigns properties to protein atoms using the updated function
    protein_atom_properties = {}
    for atom in protein_atoms:
        properties = assign_protein_atom_properties(atom)
        protein_atom_properties[atom] = properties

    # Initializes SIFt fingerprint dictionary
    sift_fingerprint = defaultdict(int)

    # Initializes sets for ligand pharmacophore and protein patch
    ligand_pharmacophore_atoms = set()
    protein_patch_residues = set()

    # Identifies interactions
    for i, ligand_atom in enumerate(ligand_mol.GetAtoms()):
        ligand_pos = ligand_coords[i]
        close_atoms = ns.search(tuple(ligand_pos), cutoff)
        for protein_atom in close_atoms:
            protein_pos = protein_atom.get_coord()
            distance = np.linalg.norm(ligand_pos - protein_pos)
            interaction_types = classify_interaction(
                i,
                ligand_atom_properties,
                protein_atom,
                protein_atom_properties,
                distance
            )
            for interaction_type in interaction_types:
                residue_index = get_residue_index(protein_atom.get_parent())
                # Set the corresponding bit in the fingerprint
                sift_fingerprint[(residue_index, interaction_type)] = 1
                # Collect ligand atoms and protein residues involved in interactions
                ligand_pharmacophore_atoms.add(i)
                protein_patch_residues.add(residue_index)

    # I can now use ligand_pharmacophore_atoms and protein_patch_residues for further steps
    return sift_fingerprint, ligand_atom_properties, protein_atom_properties

# Functions to assign properties to ligand atoms:
def is_hbond_donor(atom):
    # Checks if the atom is a hydrogen bonded to N, O, or S
    if atom.GetSymbol() == 'H':
        bonded_atoms = atom.GetNeighbors()
        if bonded_atoms:
            bonded_atom = bonded_atoms[0]
            if bonded_atom.GetSymbol() in ['N', 'O', 'S']:
                return True
    return False

def is_hbond_acceptor(atom):
    # Acceptors are electronegative atoms with lone pairs (I am not considering + ions)
    if atom.GetSymbol() in ['O', 'N', 'S']:
        return True
    return False

#def is_aromatic_atom(atom):
#     return atom.GetIsAromatic()

def is_aromatic_atom(atom):
    """
    Checks if an atom is aromatic using RDKit's aromaticity model
    ensures that the atom is part of an aromatic system.
    """
    # Ensure the molecule has aromaticity information computed (the old function did not computed aromaticity (they were all non-aromatic))
    mol = atom.GetOwningMol()
    Chem.SanitizeMol(mol, Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_SYMMRINGS, catchErrors=True)
    
    return atom.GetIsAromatic()


def is_hydrophobic_atom(atom):
    symbol = atom.GetSymbol()
    if symbol == 'C':
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
        if all(sym in ['C', 'H'] for sym in neighbors):
            return True
    return False

def is_positive_atom(atom):
    charge = atom.GetFormalCharge()
    if charge > 0:
        return True
    return False

def is_negative_atom(atom):
    charge = atom.GetFormalCharge()
    if charge < 0:
        return True
    return False

# Function to assign properties to protein atoms
def assign_protein_atom_properties(atom):
    element = atom.element.strip() if atom.element else atom.get_name()[0].upper()
    is_hydrogen = element == 'H'

    properties = {
        'is_hydrogen': is_hydrogen,
        'is_donor': False,
        'is_acceptor': False,
        'is_aromatic': False,
        'is_hydrophobic': False,
        'is_positive': False,
        'is_negative': False,
    }

    residue = atom.get_parent()
    residue_properties = get_residue_properties(residue)

    if is_hydrogen:
        bonded_atom = get_bonded_atom(atom, residue)
        if bonded_atom:
            bonded_element = bonded_atom.element.strip() if bonded_atom.element else bonded_atom.get_name()[0].upper()
            if bonded_element in ['N', 'O', 'S']:
                properties['is_donor'] = True
    else:
        if element in ['N', 'O', 'S']:
            properties['is_acceptor'] = True
            if has_bonded_hydrogen(atom, residue):
                properties['is_donor'] = True

    properties.update(residue_properties)
    return properties

def get_bonded_atom(hydrogen_atom, residue):
    # Find the heavy atom bonded to this hydrogen
    for atom in residue.get_atoms():
        if atom != hydrogen_atom and hydrogen_atom - atom < 1.2:  # Covalent bond distance
            return atom
    return None

def has_bonded_hydrogen(atom, residue):
    # Check if the heavy atom has bonded hydrogens
    for neighbor in residue.get_atoms():
        if neighbor != atom and neighbor.element == 'H' and atom - neighbor < 1.2:
            return True
    return False

# Function to get residue properties
def get_residue_properties(residue):
    residue_name = residue.get_resname()
    positive_residues = ['ARG', 'LYS', 'HIS']
    negative_residues = ['ASP', 'GLU']
    aromatic_residues = ['PHE', 'TYR', 'TRP', 'HIS']
    hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO', 'PHE', 'TRP']

    properties = {
        'is_positive': residue_name in positive_residues,
        'is_negative': residue_name in negative_residues,
        'is_aromatic': residue_name in aromatic_residues,
        'is_hydrophobic': residue_name in hydrophobic_residues,
    }

    return properties

# Updated classify_interaction function
def classify_interaction(ligand_atom_index, ligand_atom_properties, protein_atom, protein_atom_properties, distance):
    ligand_atom = ligand_atom_properties[ligand_atom_index]
    protein_props = protein_atom_properties[protein_atom]

    interaction_types = []

    # Hydrogen bond interactions
    if distance <= 2.5:  # Shorter distance since we're considering H atoms
        # Hydrogen from ligand to acceptor in protein
        if ligand_atom['is_hydrogen'] and protein_props['is_acceptor']:
            interaction_types.append('HBond_LigandDonor')
        # Hydrogen from protein to acceptor in ligand
        if protein_props['is_hydrogen'] and ligand_atom['is_acceptor']:
            interaction_types.append('HBond_ProteinDonor')

    # Ionic interactions
    if distance <= 4.0:
        if ligand_atom['is_positive'] and protein_props['is_negative']:
            interaction_types.append('Ionic_PositiveNegative')
        if ligand_atom['is_negative'] and protein_props['is_positive']:
            interaction_types.append('Ionic_NegativePositive')

    # Hydrophobic interactions
    if distance <= 4.5:
        if ligand_atom['is_hydrophobic'] and protein_props['is_hydrophobic']:
            interaction_types.append('Hydrophobic')

    # Aromatic interactions
    if distance <= 5.0:
        if ligand_atom['is_aromatic'] and protein_props['is_aromatic']:
            interaction_types.append('Aromatic')

    return interaction_types

def get_residue_index(residue):
    # Using residue sequence number as index
    return residue.get_id()[1]


################################    FUNCTION FOR SIMILARITY AND ANALYSIS    ##############################################################################################


# Similarity Analysis Using SIFt Fingerprints
def compute_sift_similarity(fingerprints):
    num_structures = len(fingerprints)
    similarity_matrix = np.zeros((num_structures, num_structures))
    for i in range(num_structures):
        for j in range(i, num_structures):
            fp1 = fingerprints[i]
            fp2 = fingerprints[j]
            similarity = calculate_tanimoto_with_frequency(fp1, fp2)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    return similarity_matrix

# def calculate_tanimoto(fp1, fp2):
#     # Get all keys
#     keys = set(fp1.keys()).union(fp2.keys()) #extract unique keys from both fingerprints
#     vector1 = np.array([fp1.get(k, 0) for k in keys]) # extracts the interaction, sets it to 1 if present, otherwise to 0
#     vector2 = np.array([fp2.get(k, 0) for k in keys])
#     # Compute Tanimoto coefficient
#     intersection = np.sum(vector1 * vector2)
#     union = np.sum((vector1 + vector2) > 0)
#     if union == 0:
#         return 0.0
#     return intersection / union

def calculate_tanimoto_with_frequency(fp1, fp2):
    # Get all keys
    keys = set(fp1.keys()).union(fp2.keys())
    vector1 = np.array([fp1.get(k, 0) for k in keys])  # Use the actual frequency
    vector2 = np.array([fp2.get(k, 0) for k in keys])  # Use the actual frequency
    
    # Compute Tanimoto coefficient
    intersection = np.sum(np.minimum(vector1, vector2))  # Sum of minimum values for each feature
    union = np.sum(np.maximum(vector1, vector2))  # Sum of maximum values for each feature
    
    if union == 0:
        return 0.0
    return intersection / union

def cluster_ligands(similarity_matrix, labels):
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    # Condense the distance matrix
    condensed_distance = squareform(distance_matrix, checks=False) #converts to vector
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method='ward')
    dendrogram(linkage_matrix, labels=labels, color_threshold= 8)
    plt.title('Ligand Clustering Based on SIFt Similarity')
    plt.xlabel('Ligand')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()






################################    FUNCTION FOR RETREIVEING PROTEINS' FAMILIES    ##############################################################################################


def fetch_protein_family(pdb_id):
    """
    Fetch protein family information for a given PDB ID.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        protein_family = data.get("struct_keywords", {}).get("pdbx_keywords", None)
        return protein_family
    else:
        print(f"Error fetching protein family for {pdb_id}: {response.status_code}")
        return None
# Parallelized fetching of protein families
def fetch_protein_families_parallel(prepared_structures, max_workers=10):
    """
    Fetch protein family annotations concurrently for all structures.
    """
    def fetch_and_assign(structure):
        pdb_id = structure['id']
        family = fetch_protein_family(pdb_id)
        return pdb_id, family

    protein_family_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_structure = {executor.submit(fetch_and_assign, structure): structure for structure in prepared_structures}
        for future in as_completed(future_to_structure):
            structure = future_to_structure[future]
            try:
                pdb_id, family = future.result()
                protein_family_map[pdb_id] = family
                structure['protein_family'] = family
            except Exception as e:
                print(f"Error fetching family for {structure['id']}: {e}")
    return protein_family_map


# Step 1: Fetch protein family annotations concurrently
def fetch_protein_families_parallel(prepared_structures, max_workers=10):
    """
    Fetch protein family annotations concurrently for all structures.
    """
    def fetch_and_assign(structure):
        pdb_id = structure['id']
        family = fetch_protein_family(pdb_id)
        return pdb_id, family

    protein_family_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_structure = {executor.submit(fetch_and_assign, structure): structure for structure in prepared_structures}
        for future in as_completed(future_to_structure):
            structure = future_to_structure[future]
            try:
                pdb_id, family = future.result()
                protein_family_map[pdb_id] = family
                structure['protein_family'] = family
            except Exception as e:
                print(f"Error fetching family for {structure['id']}: {e}")
    return protein_family_map



# Step 2: Group similar protein families
def group_protein_families(protein_family_map):
    """
    Group similar protein families using textual similarity.
    """
    families = list(set(protein_family_map.values()))
    vectorizer = TfidfVectorizer().fit_transform(families)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, linkage="average").fit(vectorizer.toarray())
    clustered_families = defaultdict(list)

    for family, cluster_id in zip(families, clustering.labels_):
        clustered_families[cluster_id].append(family)

    # Map original families to grouped clusters
    family_to_cluster = {}
    for cluster_id, cluster_families in clustered_families.items():
        representative = ", ".join(cluster_families)
        for family in cluster_families:
            family_to_cluster[family] = representative

    return family_to_cluster


def create_tsne_plot_with_separate_legend(family_clusters):
    all_features = []
    all_labels = []

    # Collect all fingerprints and their families
    all_keys = set()
    for fingerprints in family_clusters.values():
        for fp in fingerprints:
            all_keys.update(fp.keys())
    all_keys = sorted(all_keys)  # Ensure consistent ordering

    # Convert fingerprints to dense vectors
    for family, fingerprints in family_clusters.items():
        for fp in fingerprints:
            feature_vector = np.array([fp.get(key, 0) for key in all_keys])
            all_features.append(feature_vector)
            all_labels.append(family)

    # Convert features to NumPy array
    all_features = np.array(all_features)

    # Set perplexity dynamically
    n_samples = len(all_features)
    perplexity_value = min(30, n_samples - 1)  # Perplexity must be < n_samples

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    reduced_features = tsne.fit_transform(all_features)

    # Plot the t-SNE output
    unique_families = list(set(all_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_families)))
    family_to_color = {family: color for family, color in zip(unique_families, colors)}

    # Create the main t-SNE plot
    plt.figure(figsize=(20, 15))
    for family, color in family_to_color.items():
        indices = [i for i, label in enumerate(all_labels) if label == family]
        plt.scatter(
            reduced_features[indices, 0],
            reduced_features[indices, 1],
            c=[color],
            label=family,
            s=100,
            alpha=0.8
        )

    plt.title("Ligands Colored by Grouped Protein Families", fontsize=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=15)
    plt.ylabel("t-SNE Dimension 2", fontsize=15)
    plt.tight_layout()

    # Save legend as a separate figure
    legend_fig, legend_ax = plt.subplots(figsize=(8, 12))
    legend_ax.axis("off")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=family)
        for family, color in family_to_color.items()
    ]
    legend_ax.legend(handles=legend_handles, title="Protein Families", loc="center", fontsize=10)
    legend_ax.set_title("Legend for Protein Families", fontsize=15)

    plt.show()
    return legend_fig


def create_interactive_tsne_plot(family_clusters, prepared_structures):
    """
    Create an interactive t-SNE plot using Plotly.
    Each protein family will have a different color, and the legend will be scrollable.
    """
    all_features = []
    all_labels = []
    all_ligand_names = []  # Store ligand/complex names

    # Collect all fingerprints and their families
    all_keys = set()
    for fingerprints in family_clusters.values():
        for fp in fingerprints:
            all_keys.update(fp.keys())
    all_keys = sorted(all_keys)  # Ensure consistent ordering

    # Convert fingerprints to dense vectors
    structure_index = 0
    for family, fingerprints in family_clusters.items():
        # Extract only the first name of the protein family
        short_family_name = family.split(",")[0] if family else "Unknown"
        for fp in fingerprints:
            feature_vector = [fp.get(key, 0) for key in all_keys]
            all_features.append(feature_vector)
            all_labels.append(short_family_name)  # Use the short family name
            # Use the name of the structure (complex) from prepared_structures
            all_ligand_names.append(prepared_structures[structure_index]['id'])
            structure_index += 1

    # Create a Pandas DataFrame for easier manipulation
    data = pd.DataFrame(all_features)
    data['Family'] = all_labels
    data['Ligand'] = all_ligand_names  # Use complex names as ligand names

    # Reduce dimensions using t-SNE
    from sklearn.manifold import TSNE
    if len(all_features) < 42:
        perplexity_value = max(1, len(all_features) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
       
    else:
        tsne = TSNE(n_components=2, random_state=42)

    tsne_results = tsne.fit_transform(data.iloc[:, :-2])

    # Add t-SNE results back to the DataFrame
    data['Dimension 1'] = tsne_results[:, 0]
    data['Dimension 2'] = tsne_results[:, 1]

    # Create an interactive Plotly scatter plot
    fig = px.scatter(
        data,
        x='Dimension 1',
        y='Dimension 2',
        color='Family',
        title='Ligands Colored by Grouped Protein Families',
        hover_data={'Family': True, 'Ligand': True},  # Include ligand/complex names in hover
    )

    fig.update_layout(
        legend_title='Protein Families',
        #showlegend=False,
        legend=dict(
            itemsizing='constant',  # Optimize spacing
            title_font_size=10,
            font=dict(size=8),
            x=1.05,  # Push the legend outside the plot area
            y=1,  # Align legend to the top
            tracegroupgap=5,  # Adjust spacing between groups in the legend
        ),
        margin=dict(l=40, r=200, t=60, b=40),  # Add more margin to the right for the legend
    )

    # Show the plot
    fig.show()


######################    ATOMIC PHARMACOPHORE  #######################################################################################

def create_ligand_pharmacophore(ligand_mol, ligand_atom_properties):
    """
    Create a pharmacophore object for the ligand based on its atom properties.
    """
    pharmacophore = {"hydrogen_bond_donors": [],
                     "hydrogen_bond_acceptors": [],
                     "aromatic_rings": [],
                     "hydrophobic_atoms": [],
                     "positive_atoms": [],
                     "negative_atoms": []}
    
    aromatic_atoms = set()  # Track all aromatic atoms

    for atom_idx, properties in ligand_atom_properties.items():
        if properties["is_donor"]:
            pharmacophore["hydrogen_bond_donors"].append(atom_idx)
        if properties["is_acceptor"]:
            pharmacophore["hydrogen_bond_acceptors"].append(atom_idx)
        if properties["is_aromatic"]:
            pharmacophore["aromatic_rings"].append(atom_idx)
            aromatic_atoms.add(atom_idx)  # Add to aromatic atoms set
        if properties["is_hydrophobic"]:
            pharmacophore["hydrophobic_atoms"].append(atom_idx)
        if properties["is_positive"]:
            pharmacophore["positive_atoms"].append(atom_idx)
        if properties["is_negative"]:
            pharmacophore["negative_atoms"].append(atom_idx)


        # Remove aromatic atoms from the hydrophobic list
    pharmacophore["hydrophobic_atoms"] = [
        idx for idx in pharmacophore["hydrophobic_atoms"] if idx not in aromatic_atoms
    ]

    return pharmacophore


def create_protein_pharmacophore(protein_atoms, protein_atom_properties):
    """
    Create a pharmacophore object for the protein based on its atom properties.
    """
    pharmacophore = {"hydrogen_bond_donors": [],
                     "hydrogen_bond_acceptors": [],
                     "hydrophobic_residues": [],
                     "aromatic_residues": []}
    
    for atom in protein_atoms:
        properties = protein_atom_properties[atom]
        if properties["is_donor"]:
            pharmacophore["hydrogen_bond_donors"].append(atom)
        if properties["is_acceptor"]:
            pharmacophore["hydrogen_bond_acceptors"].append(atom)
        if properties["is_hydrophobic"]:
            pharmacophore["hydrophobic_residues"].append(atom.get_parent())
        if properties["is_aromatic"]:
            pharmacophore["aromatic_residues"].append(atom.get_parent())
    
    return pharmacophore


from py3Dmol import view as p3d_view


def visualize_pharmacophores(ligand_mol, ligand_pharmacophore, protein_structure, protein_pharmacophore):
    """
    Visualize pharmacophores of both ligand and protein using PyMOL.
    This function will:
    - Write ligand and protein to PDB.
    - Load them into PyMOL.
    - Represent ligand as gray sticks and protein as white sticks.
    - Add pharmacophore spheres for ligand and protein with corresponding colors.
    - Create a PNG image of the final scene and open it.
    """

    # Save ligand and protein to PDB
    ligand_pdb_file = "temp_ligand.pdb"
    save_ligand_to_pdb(ligand_mol, ligand_pdb_file)

    protein_pdb_file = "temp_protein.pdb"
    save_protein_to_pdb(protein_structure, protein_pdb_file)

    # Launch PyMOL with GUI (remove '-c' for GUI)
    # If you want no GUI, use: pymol.finish_launching(['pymol','-cq'])
    pymol.finish_launching()

    # Load the structures
    cmd.load(ligand_pdb_file, "ligand")
    cmd.load(protein_pdb_file, "protein")

    # Hide everything, then show sticks for ligand and protein
    cmd.hide("everything")
    cmd.show("sticks", "ligand")
    cmd.show("sticks", "protein")
    cmd.color("gray", "ligand")
    cmd.color("white", "protein")

    # Function to add a pseudoatom as a sphere
    def add_sphere(coords, color, radius, base_name):
        # Create a unique name for the pseudoatom
        sphere_name = f"{base_name}_{coords[0]}_{coords[1]}_{coords[2]}"
        cmd.pseudoatom(sphere_name, pos=coords)
        cmd.show("spheres", sphere_name)
        cmd.color(color, sphere_name)
        cmd.set("sphere_scale", radius, sphere_name)
        # Set transparency (sphere_transparency works from 0 to 1)
        # original code used opacity=0.5 → transparency = 0.5
        cmd.set("sphere_transparency", 0.5, sphere_name)

    # Get ligand coordinates and add pharmacophore spheres
    ligand_conf = ligand_mol.GetConformer()

    for donor_idx in ligand_pharmacophore["hydrogen_bond_donors"]:
        pos = ligand_conf.GetAtomPosition(donor_idx)
        add_sphere((pos.x, pos.y, pos.z), "blue", 1.5, "ligand_donor")

    for acceptor_idx in ligand_pharmacophore["hydrogen_bond_acceptors"]:
        pos = ligand_conf.GetAtomPosition(acceptor_idx)
        add_sphere((pos.x, pos.y, pos.z), "red", 1.5, "ligand_acceptor")

    for hydrophobic_idx in ligand_pharmacophore["hydrophobic_atoms"]:
        pos = ligand_conf.GetAtomPosition(hydrophobic_idx)
        add_sphere((pos.x, pos.y, pos.z), "yellow", 1.5, "ligand_hydrophobic")

    for aromatic_idx in ligand_pharmacophore["aromatic_rings"]:
        pos = ligand_conf.GetAtomPosition(aromatic_idx)
        add_sphere((pos.x, pos.y, pos.z), "green", 1.5, "ligand_aromatic")

    # Protein pharmacophores are given as coordinates in lists of atoms/residues
    # Each protein pharmacophore entry has coordinates directly (as per original code)
    for donor_atom in protein_pharmacophore["hydrogen_bond_donors"]:
        add_sphere((donor_atom.coord[0], donor_atom.coord[1], donor_atom.coord[2]),
                   "blue", 2.0, "protein_donor")

    for acceptor_atom in protein_pharmacophore["hydrogen_bond_acceptors"]:
        add_sphere((acceptor_atom.coord[0], acceptor_atom.coord[1], acceptor_atom.coord[2]),
                   "red", 2.0, "protein_acceptor")

    for hydrophobic_residue in protein_pharmacophore["hydrophobic_residues"]:
        atom_coords = [atom.coord for atom in hydrophobic_residue]
        center = tuple(sum(x) / len(atom_coords) for x in zip(*atom_coords))
        add_sphere(center, "yellow", 2.5, "protein_hydrophobic")

    for aromatic_residue in protein_pharmacophore["aromatic_residues"]:
        atom_coords = [atom.coord for atom in aromatic_residue]
        center = tuple(sum(x) / len(atom_coords) for x in zip(*atom_coords))
        add_sphere(center, "green", 3.0, "protein_aromatic")

    # Center and zoom the view
    cmd.zoom("all")
    cmd.bg_color("white")

    # Render a PNG image
    image_filename = "pharmacophores.png"
    cmd.png(image_filename, width=600, height=600, dpi=300, ray=1)

    # Open the PNG image
    absolute_path = os.path.abspath(image_filename)
    webbrowser.open("file://" + absolute_path)



def visualize_ligand(ligand_mol, ligand_pharmacophore, protein_structure):
    """
    Visualize ligand pharmacophores using PyMOL.
    Similar to the above but only ligand and protein are shown,
    and ligand pharmacophore spheres are added.
    """

    ligand_pdb_file = "temp_ligand.pdb"
    save_ligand_to_pdb(ligand_mol, ligand_pdb_file)

    protein_pdb_file = "temp_protein.pdb"
    save_protein_to_pdb(protein_structure, protein_pdb_file)

    pymol.finish_launching()

    # Load structures
    cmd.load(ligand_pdb_file, "ligand")
    cmd.load(protein_pdb_file, "protein")

    cmd.hide("everything")
    cmd.show("sticks", "ligand")
    cmd.color("gray", "ligand")

    cmd.show("sticks", "protein")
    cmd.color("white", "protein")

    def add_sphere(coords, color, radius=1.5):
        sphere_name = f"ligand_pharma_{coords[0]}_{coords[1]}_{coords[2]}"
        cmd.pseudoatom(sphere_name, pos=coords)
        cmd.show("spheres", sphere_name)
        cmd.color(color, sphere_name)
        cmd.set("sphere_scale", radius, sphere_name)
        cmd.set("sphere_transparency", 0.5, sphere_name)

    ligand_conf = ligand_mol.GetConformer()
    for donor_idx in ligand_pharmacophore["hydrogen_bond_donors"]:
        pos = ligand_conf.GetAtomPosition(donor_idx)
        add_sphere((pos.x, pos.y, pos.z), "blue")

    for acceptor_idx in ligand_pharmacophore["hydrogen_bond_acceptors"]:
        pos = ligand_conf.GetAtomPosition(acceptor_idx)
        add_sphere((pos.x, pos.y, pos.z), "red")

    for hydrophobic_idx in ligand_pharmacophore["hydrophobic_atoms"]:
        pos = ligand_conf.GetAtomPosition(hydrophobic_idx)
        add_sphere((pos.x, pos.y, pos.z), "yellow")

    for aromatic_idx in ligand_pharmacophore["aromatic_rings"]:
        pos = ligand_conf.GetAtomPosition(aromatic_idx)
        add_sphere((pos.x, pos.y, pos.z), "green", radius=2.0)

    # Center and zoom
    cmd.zoom("all")
    cmd.bg_color("white")

    image_filename = "ligand_pharmacophores.png"
    cmd.png(image_filename, width=600, height=600, dpi=300, ray=1)

    absolute_path = os.path.abspath(image_filename)
    webbrowser.open("file://" + absolute_path)

# Example Usage
#ligand_pharmacophore = create_ligand_pharmacophore(ligand_0, ligand_atom_properties)
#protein_pharmacophore = create_protein_pharmacophore(protein_atoms, protein_atom_properties)
#pharmacophore_view = visualize_pharmacophores(ligand_0, ligand_pharmacophore, pock_0, protein_pharmacophore)
#pharmacophore_view.show()


def visualize_only_ligand_pharacophore(ligand_mol, ligand_pharmacophore, protein_structure):
    
    
# Save ligand and protein to PDB files
    ligand_pdb_file = "temp_ligand.pdb"
    save_ligand_to_pdb(ligand_mol, ligand_pdb_file)
    
    protein_pdb_file = "temp_protein.pdb"
    save_protein_to_pdb(protein_structure, protein_pdb_file)

    # Initialize py3Dmol viewer
    v = p3d_view()
    
    # Load and visualize the ligand
    #with open(ligand_pdb_file, "r") as file:
        #v.addModel(file.read(), "pdb")
    #v.setStyle({"model": 0}, {"stick": {"color": "gray"}})

    # Add pharmacophore spheres for the ligand
    def add_sphere(coord, color, radius=1.5):
        v.addSphere({
            "center": {"x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])},
            "radius": radius,
            "color": color,
            "opacity": 0.5  # Semi-transparent
        })

    ligand_conformer = ligand_mol.GetConformer()
    for donor_idx in ligand_pharmacophore["hydrogen_bond_donors"]:
        coord = ligand_conformer.GetAtomPosition(donor_idx)
        add_sphere((coord.x, coord.y, coord.z), "blue")

    for acceptor_idx in ligand_pharmacophore["hydrogen_bond_acceptors"]:
        coord = ligand_conformer.GetAtomPosition(acceptor_idx)
        add_sphere((coord.x, coord.y, coord.z), "red")

    for hydrophobic_idx in ligand_pharmacophore["hydrophobic_atoms"]:
        coord = ligand_conformer.GetAtomPosition(hydrophobic_idx)
        add_sphere((coord.x, coord.y, coord.z), "yellow")

    for aromatic_idx in ligand_pharmacophore["aromatic_rings"]:
        coord = ligand_conformer.GetAtomPosition(aromatic_idx)
        add_sphere((coord.x, coord.y, coord.z), "green", radius = 1.5)
    
    # Center the view
    v.zoomTo()
    return v



######################################  CENTROID PHARMACOPHORE  ##########################################################################


## FUNCTIONING 

def create_ligand_pharmacophore_c(ligand_mol, ligand_atom_properties):
    """
    Create a pharmacophore object for the ligand based on atom properties.
    Functional groups, aromatic rings, and clusters are treated as single entities.
    """
    pharmacophore = {
        "hydrogen_bond_donors": [],
        "hydrogen_bond_acceptors": [],
        "aromatic_rings": [],
        "hydrophobic_features": [],
        "positive_features": [],
        "negative_features": []
    }

    # Initialize aromatic ring detection
    aromatic_rings = []

    # RDKit function for finding ring systems
    ring_info = ligand_mol.GetRingInfo()
    for ring in ring_info.AtomRings():  # Correctly use AtomRings() for atom indices in rings
        if all(ligand_atom_properties[idx]['is_aromatic'] for idx in ring):
            # Identify the center of the aromatic ring
            atom_coords = [ligand_mol.GetConformer().GetAtomPosition(idx) for idx in ring]
            center = np.mean([[coord.x, coord.y, coord.z] for coord in atom_coords], axis=0)
            aromatic_rings.append(center)

    # Adds aromatic rings to pharmacophore
    pharmacophore["aromatic_rings"].extend(aromatic_rings)

    # Classification of atoms and groups by properties
    atom_groups = {
        "hydrogen_bond_donors": [],
        "hydrogen_bond_acceptors": [],
        "hydrophobic_features": [],
        "positive_features": [],
        "negative_features": []
    }

    ligand_conformer = ligand_mol.GetConformer()
    for atom_idx, properties in ligand_atom_properties.items():
        coord = ligand_conformer.GetAtomPosition(atom_idx)
        atom_coords = np.array([coord.x, coord.y, coord.z])
        if properties["is_donor"]:
            atom_groups["hydrogen_bond_donors"].append(atom_coords)
        if properties["is_acceptor"]:
            atom_groups["hydrogen_bond_acceptors"].append(atom_coords)
        if properties["is_hydrophobic"] and not properties["is_aromatic"]:
            atom_groups["hydrophobic_features"].append(atom_coords)
        if properties["is_positive"]:
            atom_groups["positive_features"].append(atom_coords)
        if properties["is_negative"]:
            atom_groups["negative_features"].append(atom_coords)

    # Cluster groups to create pharmacophore features
    def cluster_features(features, cluster_threshold=3):
        """
        Cluster features into single pharmacophore points.
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import fcluster, linkage

        if len(features) <= 2:
            return features

        # Compute distance matrix
        dist_matrix = squareform(pdist(features))
        # Perform clustering
        clusters = fcluster(linkage(dist_matrix, method='average'), cluster_threshold, criterion='distance')

        # Create cluster centers
        cluster_centers = []
        for cluster_id in set(clusters):
            cluster_points = np.array([features[i] for i in range(len(features)) if clusters[i] == cluster_id])
            cluster_centers.append(np.mean(cluster_points, axis=0))

        return cluster_centers

    # Add clustered features to the pharmacophore
    for key in atom_groups.keys():
        clustered_features = cluster_features(atom_groups[key])
        pharmacophore[key].extend(clustered_features)

    return pharmacophore

# functioning function

from py3Dmol import view as p3d_view

# def visualize_ligand_pharmacophore2(ligand_mol, pharmacophore):
#     """
#     Visualize the ligand and its pharmacophore using PyMOL.
#     This version groups pseudoatoms by feature type to ensure spheres are visible.
#     """

#     # Save ligand to PDB
#     ligand_pdb_file = "temp_ligand.pdb"
#     Chem.MolToPDBFile(ligand_mol, ligand_pdb_file)

#     # Launch PyMOL (GUI mode)
#     # For headless: 
#     pymol.finish_launching(['pymol','-q'])
#     #pymol.finish_launching()

#     # Load the ligand
#     cmd.load(ligand_pdb_file, "ligand")
#     cmd.hide("everything", "all")
#     cmd.show("sticks", "ligand")
#     cmd.color("gray", "ligand")

#     # Feature type to color mapping
#     feature_colors = {
#         "hydrogen_bond_donors": "blue",
#         "hydrogen_bond_acceptors": "red",
#         "aromatic_rings": "green",
#         "hydrophobic_features": "yellow",
#         "positive_features": "magenta",
#         "negative_features": "cyan"
#     }

#     # Radius and transparency
#     sphere_radius = 1.5
#     sphere_transparency = 0.5

#     # For each feature type, create a separate object and add all pseudoatoms to it
#     for feature_type, features_list in pharmacophore.items():
#         if not features_list:
#             continue
#         color = feature_colors.get(feature_type, "white")
#         object_name = f"pharma_{feature_type}"
        
#         # Create the first pseudoatom to initialize the object
#         first_coord = tuple(float(x) for x in features_list[0])
#         cmd.pseudoatom(object_name, pos=first_coord)
        
#         # Add the remaining pseudoatoms to the same object
#         for feature in features_list[1:]:
#             coords = tuple(float(x) for x in feature)
#             # Add another pseudoatom to the same object
#             cmd.pseudoatom(object=object_name, pos=coords)

#         # Now show these pseudoatoms as spheres
#         cmd.show("spheres", object_name)
#         cmd.color(color, object_name)
#         cmd.set("sphere_scale", sphere_radius, object_name)
#         cmd.set("sphere_transparency", sphere_transparency, object_name)

#     # Adjust view
#     cmd.bg_color("white")
#     cmd.zoom("all")
#     input("press enter to close...")
#     # Render a PNG
#     image_filename = "ligand_pharmacophore.png"
#     cmd.png(image_filename, width=800, height=600, dpi=300, ray=1)

#     # Open the resulting image
#     absolute_path = os.path.abspath(image_filename)
#     webbrowser.open("file://" + absolute_path)

def visualize_ligand_pharmacophore2(ligand_mol, pharmacophore):
    """
    Visualize the ligand and its pharmacophore using PyMOL.
    This version groups pseudoatoms by feature type to ensure spheres are visible,
    and saves the PyMOL session to the current directory.
    """

    # Save ligand to PDB
    ligand_pdb_file = "temp_ligand.pdb"
    Chem.MolToPDBFile(ligand_mol, ligand_pdb_file)

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    # Set attributes before any QApplication is created
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    import pymol
    from pymol import cmd

    # Launch PyMOL with GUI. For headless mode, use pymol.finish_launching(['pymol','-cq'])
    pymol.finish_launching(['pymol','-cq'])

    # Load the ligand
    cmd.load(ligand_pdb_file, "ligand")
    cmd.hide("everything", "all")
    cmd.show("sticks", "ligand")
    cmd.color("gray", "ligand")

    # Feature type to color mapping
    feature_colors = {
        "hydrogen_bond_donors": "blue",
        "hydrogen_bond_acceptors": "red",
        "aromatic_rings": "green",
        "hydrophobic_features": "yellow",
        "positive_features": "magenta",
        "negative_features": "cyan"
    }

    # Radius and transparency
    sphere_radius = 1.5
    sphere_transparency = 0.5

    # For each feature type, create a separate object and add all pseudoatoms to it
    for feature_type, features_list in pharmacophore.items():
        if not features_list:
            continue
        color = feature_colors.get(feature_type, "white")
        object_name = f"pharma_{feature_type}"

        # Create pseudoatoms
        for i, feature in enumerate(features_list):
            coords = tuple(float(x) for x in feature)
            if i == 0:
                cmd.pseudoatom(object_name, pos=coords)
            else:
                cmd.pseudoatom(object=object_name, pos=coords)

        # Show these pseudoatoms as spheres
        cmd.show("spheres", object_name)
        cmd.color(color, object_name)
        cmd.set("sphere_scale", sphere_radius, object_name)
        cmd.set("sphere_transparency", sphere_transparency, object_name)

    # Adjust view
    cmd.bg_color("white")
    cmd.zoom("all")

    # Optionally render a PNG
    image_filename = "./ligand_pharmacophore.png"
    cmd.png(image_filename, width=800, height=600, dpi=300, ray=1)

    # Save the PyMOL session
    session_filename = r"C:\Users\gabri\Desktop\third semester\Python Project\python project v18/ligand_pharmacophore_session.pse"
    cmd.save(session_filename)

    # If you don't need the prompt, you can remove it
    # input("press enter to close...")

    # Instead of opening the PNG automatically, we can just leave it saved
    # If you want to open it, uncomment the lines below:
    # absolute_path = os.path.abspath(image_filename)
    # webbrowser.open("file://" + absolute_path)
