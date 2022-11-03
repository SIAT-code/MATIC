import numpy as np
import sys
import rdkit
from argparse import ArgumentParser
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import sascorer as sascorer
import rdkit.Chem.QED as QED
import numpy as np

parser = ArgumentParser()
parser.add_argument('--ref_path', required=True)
args = parser.parse_args()

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

pred_data = [line.split()[:] for line in sys.stdin]
# print(pred_data[0])
# pred_data = [line.split()[1:] for line in sys.stdin]
# pred_mols = [mol for mol,x,y,qed,sa in pred_data if float(x) >= 0.5 and float(y) >= 0.5 and float(qed) > 0.6 and float(sa) < 4]
# pred_mols = [mol for mol,x,y,qed,sa in pred_data if float(x) >= 0.5 and float(y) >= 0.5 and float(qed) > 0.4 and float(sa) < 6]
pred_mols = [mol for mol,x,y in pred_data if float(x) >= 0.5 and float(y) >= 0.5]
# s_qed = []
# s_sa = []
# for mol,x,y,qed,sa in pred_data:
#     if float(x) >= 0.5 and float(y) >= 0.5 and float(qed) > 0.4 and float(sa) < 6:
#         s_qed.append(float(qed))
#         s_sa.append(float(sa))
# print('qed sa:',np.array(s_qed).mean(),np.array(s_sa).mean())

fraction_actives = len(pred_mols) / len(pred_data)
print('fraction actives:', fraction_actives)

with open(args.ref_path) as f:
    next(f)
    true_mols = [line.split(',')[0] for line in f]
print('number of active reference', len(true_mols))

true_mols = [Chem.MolFromSmiles(s) for s in true_mols]
true_mols = [x for x in true_mols if x is not None]
true_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in true_mols]

pred_mols = [Chem.MolFromSmiles(s) for s in pred_mols]
pred_mols = [x for x in pred_mols if x is not None]
pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]

fraction_similar = 0
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
    # if max(sims) >= 0.2:
    #     fraction_similar += 1
    if max(sims) >= 0.3:
        fraction_similar += 1

print('novelty:', 1 - fraction_similar / len(pred_mols))

similarity = 0
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
    similarity += sum(sims)

n = len(pred_fps) 
n_pairs = n * (n - 1) / 2
diversity = 1 - similarity / n_pairs
print('diversity:', diversity)

