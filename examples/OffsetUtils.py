import numpy as np


def SurfEdgePtsMapping(Surfaces, SurfNormals, SurfPtIdx, evalPtsSizeU, evalPtsSizeV):
    SurfEdgePtsMap = np.zeros([len(Surfaces), evalPtsSizeU * evalPtsSizeV, 12], dtype=np.uint32)

    for i in range(len(Surfaces)):
        for j in range(evalPtsSizeU * evalPtsSizeV):
            if SurfPtIdx[i][j]:
                count = 0
                SurfEdgePtsMap[i][j][0] = count

                for k in range(len(Surfaces)):
                    if i != k:
                        for l in range(evalPtsSizeU * evalPtsSizeV):
                            if SurfPtIdx[k][l]:
                                if np.linalg.norm(np.array(Surfaces[i].evalpts[j]) -
                                                  np.array(Surfaces[k].evalpts[l])) < 1e-3:
                                    count += 1
                                    SurfEdgePtsMap[i][j][0] = count
                                    SurfEdgePtsMap[i][j][(2 * SurfEdgePtsMap[i][j][0]) - 1] = k
                                    SurfEdgePtsMap[i][j][(2 * SurfEdgePtsMap[i][j][0]) - 0] = l

    np.save('StabilizerSurfEdgePtMap_%d_size.npy' % evalPtsSizeV, SurfEdgePtsMap)

    print('File Save Done !!')

    # SurfEdgePtsMap = np.load('AortaSurfEdgePtMap_%d_size.npy' % evalPtsSize)

    for i in range(len(Surfaces)):
        for j in range(evalPtsSizeU * evalPtsSizeV):
            if SurfEdgePtsMap[i][j][0] != 0:
                temp = np.zeros(3)
                temp += SurfNormals[i][j]

                for k in range(SurfEdgePtsMap[i][j][0]):
                    SurfIdx = SurfEdgePtsMap[i][j][(2 * k) + 1]
                    PtIdx = SurfEdgePtsMap[i][j][(2 * k) + 2]

                    temp += SurfNormals[SurfIdx][PtIdx]
                    temp_norm = np.linalg.norm(temp)
                    if temp_norm != 0:
                        temp /= temp_norm

                SurfNormals[i][j] = temp
                for k in range(SurfEdgePtsMap[i][j][0]):
                    SurfIdx = SurfEdgePtsMap[i][j][(2 * k) + 1]
                    PtIdx = SurfEdgePtsMap[i][j][(2 * k) + 2]
                    SurfNormals[SurfIdx][PtIdx] = temp

    return SurfNormals


def MapEdgeCtrlPts(Surfs, EdgePtsIdx):
    for i in range(len(Surfs)):
        for j in range(len(Surfs)):

            if i != j:
                for k in range(EdgePtsIdx[i].shape[0]):
                    Pt1 = np.array(Surfs[i].ctrlpts[EdgePtsIdx[i][k][0]])
                    for l in range(EdgePtsIdx[j].shape[0]):
                        Pt2 = np.array(Surfs[j].ctrlpts[EdgePtsIdx[j][l][0]])

                        if np.linalg.norm(Pt1 - Pt2) < 0.001:
                            EdgePtsIdx[i][k][1] += 1
                            if EdgePtsIdx[i][k][1] == 6:
                                print("5 common")

                            EdgePtsIdx[i][k][2 * EdgePtsIdx[i][k][1] + 0] = j
                            EdgePtsIdx[i][k][2 * EdgePtsIdx[i][k][1] + 1] = EdgePtsIdx[j][l][0]
    return EdgePtsIdx


def EdgeCtrlPtsIndexing(surf, EdgePtsCount):
    MapSize = 12
    EdgePtsIdx = np.empty([EdgePtsCount, MapSize], dtype=np.int16)
    count = 0

    for i in range(0, len(surf.ctrlpts)):
        AdjPts = np.empty(4, dtype=np.int16)

        AdjPts[0] = i + surf.ctrlpts_size_v
        AdjPts[1] = i + 1
        AdjPts[2] = i - surf.ctrlpts_size_v
        AdjPts[3] = i - 1

        if AdjPts[0] >= surf.cpsize[0] * surf.cpsize[1]:
            AdjPts[0] = -1
        if AdjPts[1] == surf.cpsize[1] * ((i // surf.cpsize[1]) + 1):
            AdjPts[1] = -1
        if AdjPts[2] < 0:
            AdjPts[2] = -1
        if AdjPts[3] == (surf.cpsize[1] * (i // surf.cpsize[1])) - 1:
            AdjPts[3] = -1

        if np.any(AdjPts == -1):
            EdgePtsIdx[count, 0] = i
            EdgePtsIdx[count, 1] = 0
            EdgePtsIdx[count, 2:MapSize] = -1
            count += 1

    return EdgePtsIdx
    pass
