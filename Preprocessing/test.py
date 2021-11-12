import os
import matplotlib.pyplot as plt


def concatStr(str_list):
  return ''.join(str_list)


if __name__ == "__main__":
    rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/"
    plotPath = "/Users/mainroot/RPC_graph/ML_V1/ML_V1_AfterDBSCAN/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    filesizes = []
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        for dpidName in dpidNames:
            filesize = os.path.getsize(concatStr([rpcImonSepPath, dpidName]))
            filesizes.append(filesize)

    print(filesizes)


    plt.hist(filesizes, bins=100)
    plt.show()