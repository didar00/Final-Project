
import chemprop

# read dataset file
# create a data (molecule) point for each
# smiles representation

with open("chemprop_data/bbbp.csv", "r") as f: # make this automatic
    lines = f.readlines()[1:]

lines = list(map(str.strip, lines))

print("LINES")
print(lines[0:10])
print()

molecule_list = list()

for line in lines:
    rep, target = line.split(",")[0], float(line.split(",")[1])
    temp = list()
    temp.append(rep)


    #molecule = chemprop.data.data.MoleculeDatapoint(smiles=list(rep), targets=ind)
    molecule = chemprop.data.data.MoleculeDatapoint(smiles=temp, targets=target)
    molecule_list.append(molecule)

print("MOLECULES")
print(molecule_list[0:10])
print()

# load molecule list to the molecule dataset
# to use it in the scaffold split later
molecule_obj = chemprop.data.data.MoleculeDataset(molecule_list)

# time to split the dataset using scaffold split
(train, val, test) = chemprop.data.scaffold.scaffold_split(data = molecule_obj, sizes = (0.8, 0.1, 0.1), seed = 42)

# it should be ready already
# just tset it

print(val.atom_features_size())
print(val.smiles())
print("************************************")
print(test.targets())

