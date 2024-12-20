import argparse
import os
from collections import defaultdict
from rdkit import Chem
import webbrowser
from Bio.PDB import Selection
from .python_interactions_library import (
    load_pdbbind_refined_data,
    fetch_all_resolutions,
    prepare_structures,
    determine_sift_interactions,
    compute_sift_similarity,
    cluster_ligands,
    fetch_protein_families_parallel,
    group_protein_families,
    create_interactive_tsne_plot,
    create_ligand_pharmacophore_c,
    create_protein_pharmacophore
)

def visualize_ligand_pharmacophore2(ligand_mol, pharmacophore):
    """
    Visualize the ligand and its pharmacophore using PyMOL.
    This version groups pseudoatoms by feature type to ensure spheres are visible,
    and saves the PyMOL session and a PNG image.
    """
    ligand_pdb_file = "temp_ligand.pdb"
    Chem.MolToPDBFile(ligand_mol, ligand_pdb_file)

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    import pymol
    from pymol import cmd

    # Launch PyMOL in headless mode for stability:
    pymol.finish_launching(['pymol', '-cq'])
    cmd.reinitialize()

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

    sphere_radius = 1.5
    sphere_transparency = 0.5

    # For each feature type, create a separate object and add all pseudoatoms
    for feature_type, features_list in pharmacophore.items():
        if not features_list:
            continue
        color = feature_colors.get(feature_type, "white")
        object_name = f"pharma_{feature_type}"

        for i, feature in enumerate(features_list):
            coords = tuple(float(x) for x in feature)
            if i == 0:
                cmd.pseudoatom(object_name, pos=coords)
            else:
                cmd.pseudoatom(object=object_name, pos=coords)

        cmd.show("spheres", object_name)
        cmd.color(color, object_name)
        cmd.set("sphere_scale", sphere_radius, object_name)
        cmd.set("sphere_transparency", sphere_transparency, object_name)

    cmd.bg_color("white")
    cmd.zoom("all")

    # Save a PNG image
    image_filename = "ligand_pharmacophore.png"
    cmd.png(image_filename, width=800, height=600, dpi=300, ray=1)

    # Save the PyMOL session
    session_filename = "ligand_pharmacophore_session.pse"
    cmd.save(session_filename)

    # Optional: open the PNG image
    absolute_path = os.path.abspath(image_filename)
    webbrowser.open("file://" + absolute_path)

    cmd.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Run the SIFt analysis pipeline and optionally visualize a pharmacophore."
    )
    parser.add_argument(
        "-d", "--data_dir",
        help="Path to the data directory containing the PDBBind refined structures",
        required=True
    )
    parser.add_argument(
        "-p", "--pharmacophore",
        help="Name of the complex for which to visualize and save the pharmacophore (e.g., 1a4w)",
        required=False
    )
    parser.add_argument(
        "-r", "--res_tresh",
        help="Resolution threshold in Å. Complexes with resolution above this value will be skipped.",
        type=float,
        default=2.0
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    user_complex = args.pharmacophore
    res_tresh = args.res_tresh

    # Load data
    complex_dirs = load_pdbbind_refined_data(data_dir)
    pdb_ids = [os.path.basename(d) for d in complex_dirs]
    print("Loading the dataset (can take up to 10 minutes)...")
    print("Fetching resolutions for all PDB IDs...")
    resolution_cache = fetch_all_resolutions(pdb_ids)

    print("Preparing structures...")
    prepared_structures, skipped_folders, skipped_folders_res = prepare_structures(complex_dirs, resolution_cache, res_tresh)

    print(f"\nPrepared structures: {len(prepared_structures)}")
    print("number of complexes in the dataset:", len(complex_dirs))
    print(f"number of skipped complexes (folders) because of res > {res_tresh} A : {len(skipped_folders_res)}")
    print(f"number of skipped complexes due to failed ligand kekulization (within those having res < {res_tresh} A) : {len(skipped_folders)}")
    print("number of successfully loaded complexes:", len(prepared_structures))

    complex_name_to_index = {c['id']: idx for idx, c in enumerate(prepared_structures)}

    # Determine SIFt interactions
    print("Determining SIFt interactions...")
    all_sift_fingerprints = []
    labels = []
    cutoff = 5

    for prepared_structure in prepared_structures:
        try:
            sift_fp = determine_sift_interactions(prepared_structure, cutoff)[0]
            if sift_fp:
                all_sift_fingerprints.append(sift_fp)
                labels.append(prepared_structure['id'])
            else:
                print(f"No interactions found for {prepared_structure['id']}")
        except Exception as e:
            print(f"Error processing {prepared_structure['id']}: {e}")

    print(f"Total fingerprints generated: {len(all_sift_fingerprints)}")
    for i in range(len(all_sift_fingerprints)):
        print(labels[i])
        print(all_sift_fingerprints[i])

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = compute_sift_similarity(all_sift_fingerprints)

    # Cluster ligands and visualize
    print("Clustering ligands...")
    cluster_ligands(similarity_matrix, labels)

    # Fetch protein family annotations concurrently
    print("Fetching protein family annotations concurrently...")
    protein_family_map = fetch_protein_families_parallel(prepared_structures)

    # Group fingerprints by protein family
    print("\nGrouping fingerprints by protein family...")
    family_clusters = defaultdict(list)
    ligand_to_families = defaultdict(set)  # To track ligands belonging to multiple families

    for structure, fingerprint in zip(prepared_structures, all_sift_fingerprints):
        family = structure.get('protein_family', 'Unknown')
        family_clusters[family].append(fingerprint)
        ligand_to_families[structure['id']].add(family)

    print("\nNumber of ligands per protein family:")
    for family, fingerprints in family_clusters.items():
        print(f"{family}: {len(fingerprints)} ligands")

    print("\nLigands belonging to multiple families:")
    ligands_in_multiple_families = {
        ligand: families
        for ligand, families in ligand_to_families.items()
        if len(families) > 1
    }

    if ligands_in_multiple_families:
        for ligand, families in ligands_in_multiple_families.items():
            print(f"{ligand} belongs to families: {', '.join(families)}")
    else:
        print("No ligands found in multiple families.")

    # Create t-SNE plot for ligands
    print("\nCreating t-SNE plot for ligands colored by protein family...")

    print("Fetching protein family annotations concurrently...")
    protein_family_map = fetch_protein_families_parallel(prepared_structures)

    print("\nGrouping similar protein families...")
    family_to_cluster = group_protein_families(protein_family_map)

    family_clusters = defaultdict(list)
    ligand_to_families = defaultdict(set)  # To track ligands belonging to multiple families

    for structure, fingerprint in zip(prepared_structures, all_sift_fingerprints):
        original_family = structure.get('protein_family', 'Unknown')
        grouped_family = family_to_cluster.get(original_family, 'Unknown')
        family_clusters[grouped_family].append(fingerprint)
        ligand_to_families[structure['id']].add(grouped_family)

    print("\nNumber of ligands per grouped protein family:")
    for family, fingerprints in family_clusters.items():
        print(f"{family}: {len(fingerprints)} ligands")

    print("\nLigands belonging to multiple families:")
    ligands_in_multiple_families = {
        ligand: families
        for ligand, families in ligand_to_families.items()
        if len(families) > 1
    }

    if ligands_in_multiple_families:
        for ligand, families in ligands_in_multiple_families.items():
            print(f"{ligand} belongs to families: {', '.join(families)}")
    else:
        print("No ligands found in multiple families.")

    print("\nCreating t-SNE plot for grouped protein families...")
    create_interactive_tsne_plot(family_clusters, prepared_structures)

    # If user provided a complex name for pharmacophore visualization
    if user_complex and user_complex in complex_name_to_index:
        index2 = complex_name_to_index[user_complex]
        sift_fingerprint2, ligand_atom_properties2, protein_atom_properties2 = determine_sift_interactions(prepared_structures[index2], 9)

        # Create pharmacophore objects
        ligand_pharmacophore2 = create_ligand_pharmacophore_c(prepared_structures[index2]['ligand'], ligand_atom_properties2)
        protein_pharmacophore2 = create_protein_pharmacophore(
            Selection.unfold_entities(prepared_structures[index2]['protein'], 'A'),
            protein_atom_properties2
        )

        for key, features in ligand_pharmacophore2.items():
            print(f"{key}:")
            for feature in features:
                print(f"  {feature}")

        # Visualize and save the session
        visualize_ligand_pharmacophore2(prepared_structures[index2]['ligand'], ligand_pharmacophore2)


if __name__ == "__main__":
    main()
