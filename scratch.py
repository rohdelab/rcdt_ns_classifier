def fun_classify(N_cv, dataTe, dataM, labels, indtr, template, v1, v2):
    eps = 1e-8
    dist = 10000 * np.ones([numClass, dataTe.shape[1]])
    yPred = -1 * np.ones([1, dataTe.shape[1]])

    for class_idx in range(numClass):
        ## Training phase
        cls_l = np.where(labels == class_idx)
        cls_l = cls_l[1]

        # take x_ax number of train samples
        ind = min(cls_l) + indtr[0:x_ax, N_cv] - 1
        dataTr = []

        pl_count = min(mp.cpu_count(), len(ind))
        batchTr = dataM[:, :, ind]
        pl = mp.Pool(pl_count)
        splits = np.array_split(batchTr, pl_count, axis=2)

        dataRCDT = pl.map(fun_rcdt_batch, splits)
        dataTr = np.vstack(dataRCDT)
        dataTr = dataTr.T
        pl.close()
        pl.join()

        ######## need to add deformation########
        dataTr = np.concatenate((dataTr, v1), axis=1)
        dataTr = np.concatenate((dataTr, v2), axis=1)

        # generate the bases vectors

        u, s, vh = LA.svd(dataTr)
        # choose first 512 components if train sample>512
        s_num = min(512, len(s))
        s = s[0:s_num]
        s_ind = np.where(s > eps)
        basis = u[:, s_ind[0]]

        ## Testing Phase

        proj = basis.T @ dataTe

        # dataTe: (h, N), basis: (h, M), proj: (N, M)

        projR = basis @ proj  # projR: (h, N)
        dist[class_idx] = LA.norm(projR - dataTe, axis=0)

    for i in range(dataTe.shape[1]):
        d = dist[:, i]
        yPred[0, i] = np.where(d == min(d))[0]

    return yPred
