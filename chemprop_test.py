

def split_data(filename):
    import chemprop

    # read dataset file
    # create a data (molecule) point for each
    # smiles representation

    with open(filename, "r") as f: # make this automatic
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

    print(val.atom_features_size())
    print(val.smiles())
    print("************************************")
    print(test.targets())

    return (train, val, test)

split_data("chemprop_data/classification/bace.csv")