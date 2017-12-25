import PCA_tSNE_MDS as ptm

HISorHS = "HS"
file = "~/Dropbox/missense_pred/data/Ben/input_data." + HISorHS + ".csv"
supTrain = False

ptm.PCAanalys(file, "layer1", HISorHS, False, "analysis/PCA/PCALayer1" + HISorHS + "unsup")
ptm.PCAanalys(file, "layer2", HISorHS, False, "analysis/PCA/PCALayer2" + HISorHS + "unsup")
ptm.PCAanalys(file, "layer3", HISorHS, False, "analysis/PCA/PCALayer3" + HISorHS + "unsup")

ptm.PCAanalys(file, "layer1", HISorHS, True, "analysis/PCA/PCALayer1" + HISorHS + "sup")
ptm.PCAanalys(file, "layer2", HISorHS, True, "analysis/PCA/PCALayer2" + HISorHS + "sup")
ptm.PCAanalys(file, "layer3", HISorHS, True, "analysis/PCA/PCALayer3" + HISorHS + "sup")

ptm.tSNEanalys(file, "layer1", HISorHS, False, "analysis/t-SNE/t-SNELayer1" + HISorHS + "unsup")
ptm.tSNEanalys(file, "layer2", HISorHS, False, "analysis/t-SNE/t-SNELayer2" + HISorHS + "unsup")
ptm.tSNEanalys(file, "layer2", HISorHS, False, "analysis/t-SNE/t-SNELayer3" + HISorHS + "unsup")

ptm.tSNEanalys(file, "layer1", HISorHS, True, "analysis/t-SNE/t-SNELayer1" + HISorHS + "sup")
ptm.tSNEanalys(file, "layer2", HISorHS, True, "analysis/t-SNE/t-SNELayer2" + HISorHS + "sup")
ptm.tSNEanalys(file, "layer2", HISorHS, True, "analysis/t-SNE/t-SNELayer3" + HISorHS + "sup")

ptm.MDSanalys(file, "layer1", HISorHS, False, "analysis/MDS/MDSLayer1" + HISorHS + "unsup")
ptm.MDSanalys(file, "layer2", HISorHS, False, "analysis/MDS/MDSLayer2" + HISorHS + "unsup")
ptm.MDSanalys(file, "layer3", HISorHS, False, "analysis/MDS/MDSLayer3" + HISorHS + "unsup")

ptm.MDSanalys(file, "layer1", HISorHS, True, "analysis/MDS/MDSLayer1" + HISorHS + "sup")
ptm.MDSanalys(file, "layer2", HISorHS, True, "analysis/MDS/MDSLayer2" + HISorHS + "sup")
ptm.MDSanalys(file, "layer3", HISorHS, True, "analysis/MDS/MDSLayer3" + HISorHS + "sup")

