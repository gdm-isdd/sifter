Sifter 0.1

Installation Requirements:
Conda

Installation guide
Install the python_interactions3 with conda:
conda env create -f python_interactions3.yml
conda activate python_interactions3

Install sifter: from the environment, in the directory of sifter0.1:

python setup.py install

User Guide: 
-d : dire

Sifter: Structural Interaction Fingerprinting Tool
Sifter is a Python-based tool for analyzing and visualizing pharmacophores and structural interaction fingerprints (SIFts). It supports processing PDBBind data, visualizing pharmacophore interactions, clustering ligands based on similarity, and grouping protein families.
Features
•	Data Preparation: Prepares PDB structures for interaction analysis.
•	Pharmacophore Visualization: Generates interactive and visual representations of pharmacophores.
•	Similarity Analysis: Computes SIFt-based similarity matrices and clusters ligands.
•	Protein Family Grouping: Groups proteins based on structural keywords.
•	Interactive Plots: Creates t-SNE plots to visualize ligand clusters.
________________________________________


Installation
It is recommended to use the python_interactions3.yml file, as it is the working environment. Using conda:
conda env create -f python_interactions3.yml

Then from the sifter0.1 folder:
python setup.py install


Dependencies
The following libraries are required and will be installed automatically:
•	rdkit-pypi
•	biopython
•	pyqt5
•	numpy
•	scipy
•	matplotlib
•	scikit-learn
•	pymol
•	plotly
Make sure you have PyMOL installed for visualization. Follow PyMOL installation instructions if required.
________________________________________





Usage
Basic Command
Run the program from the command line:
bash
Copy code
sifter -d /path/to/data_dir -p complex_name -r 2.0
Arguments
•	-d, --data_dir (Required): Path to the directory containing PDBBind refined structures.
•	-p, --pharmacophore (Optional): PDB ID of the complex to visualize.
•	-r, --res_tresh (Optional): Resolution threshold in Å (default: 2.0).
Example
bash
Copy code
sifter -d ./pdb_data -p 1a4w -r 1.5
This command will analyze structures in the ./pdb_data directory, visualize pharmacophore interactions for the 1a4w complex, and skip complexes with resolution above 1.5 Å.
________________________________________
Workflow
1.	Load Data: Reads the dataset from the specified directory.
2.	Filter by Resolution: Skips complexes above the resolution threshold.
3.	Process Structures: Identifies ligands and proteins for interaction analysis.
4.	Analyze SIFt Interactions: Computes interaction fingerprints for complexes.
5.	Cluster Ligands: Groups ligands based on similarity.
6.	Visualize Pharmacophores: Uses PyMOL to render pharmacophore features.
7.	Group Protein Families: Assigns proteins to families based on structural keywords.
8.	Interactive t-SNE Plot: Visualizes ligand clusters in 2D space.
________________________________________


Outputs
•	Pharmacophore Visualizations:
o	ligand_pharmacophore.png: Visual representation of pharmacophore interactions.
o	ligand_pharmacophore_session.pse: PyMOL session file for advanced editing.
•	Cluster Dendrograms:
o	Plots showing ligand clusters based on SIFt similarity.
•	t-SNE Plots (on web browser):
o	Interactive plots highlighting grouped protein families.
________________________________________
Advanced Features
•	Custom Visualization: Use PyMOL or Plotly for advanced manipulation of results.
•	Parallel Execution: Speeds up resolution fetching using multi-threading.
•	Extensible Framework: Modify or extend functionality via the modular codebase.
________________________________________
Contact
For questions or issues, please reach out to:
•	Author: Gabriele De Marco
•	Email: gabriele.demarco.isdd@gmail.com
________________________________________
