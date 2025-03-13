import matplotlib.pyplot as plt
import numpy as np

def corner(samples, bins=20, hist_bin_factor=1, levels=[0.5, 1, 1.5, 2], labels=None, titles=None, cmap='Blues', chist='#0072B2', cdata='#CC2900', data=None, show_contours=True, figsize=(6, 6)):
    """
    Create a corner plot for visualizing multi-dimensional data.

    Parameters:
    -----------
    samples : numpy.ndarray
        The samples to plot. Should be of shape (n_samples, n_dimensions).
    bins : int, optional
        Number of bins for the histograms. Default is 20.
    hist_bin_factor : int, optional
        Factor to multiply the number of bins for the 1D histograms. Default is 1.
    levels : list, optional
        Contour levels to plot. Default is [0.5, 1, 1.5, 2] sigma levels.
    labels : list, optional
        Labels for each dimension. Default is None.
    titles : list, optional
        Titles for each dimension. Default is None.
    cmap : str, optional
        Colormap for the 2D histograms. Default is 'Blues'.
    chist : str, optional
        Color for the 1D histograms. Default is '#0072B2'.
    cdata : str, optional
        Color for the data points. Default is '#CC2900'.
    data : numpy.ndarray, optional
        Data points to plot on the corner plot. Default is None.
    show_contours : bool, optional
        Whether to show contours on the 2D histograms. Default is True.
    figsize : tuple, optional
        Size of the figure. Default is (6, 6).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the corner plot.
    ax : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
        Array of axes objects for the subplots.
    """

    N_dim = samples.shape[1]

    # choose the correct levels
    levels = np.array(levels)
    levels = 1.0 - np.exp(-0.5 * levels ** 2)

    if labels is None:
        labels = [f'x{i}' for i in range(N_dim)]
    if titles is None:
        titles = labels

    # make a corner plot structure
    fig, ax = plt.subplots(N_dim, N_dim, figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i in range(N_dim):
        for j in range(N_dim):
            if i==j:
                # turn off ylabel and yticks
                ax[i, j].set_yticklabels([])
                ax[i, j].set_yticks([])
                ax[i, j].set_title(labels[i])
            if j > i:
                ax[i, j].axis('off')
    for i in range(N_dim):
        for j in range(i+1):
            if i>0 and j==0:
                ax[i, j].set_ylabel(labels[i])
            if i==N_dim-1 and j>=0:
                ax[i, j].set_xlabel(labels[j])
            if i != N_dim-1:
                ax[i, j].set_xticklabels([])
            if j != 0:
                ax[i, j].set_yticklabels([])
            # rotate the tick labels for the x and y axes
            ax[i, j].tick_params(axis='both', rotation=45)
            ax[i, j].locator_params(axis='both', nbins=6)
    # align the labels
    fig.align_labels()


    # plot the 1D histograms
    for i in range(N_dim):
        ax[i, i].hist(samples[:, i], bins=bins*hist_bin_factor, histtype='step', density=True, color=chist)
        ax[i, i].set_xlim(samples[:, i].min(), samples[:, i].max())

    # make the 2D histograms
    for i in range(N_dim):
        for j in range(i):
            H, xedges, yedges = np.histogram2d(samples[:, j], samples[:, i], bins=bins)
            ax[i, j].imshow(H.T, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), origin='lower', aspect='auto', cmap=cmap, vmin=0)
            ax[i, j].set_xlim(xedges[0], xedges[-1])
            ax[i, j].set_ylim(yedges[0], yedges[-1])



        # plot the contours (this section is very close to the corner python package https://corner.readthedocs.io/en/latest/)
            if show_contours:
                # get the levels for the contours
                H_flat = H.flatten()
                inds = np.argsort(H_flat)[::-1]
                H_flat = H_flat[inds]
                H_cum = np.cumsum(H_flat)
                H_cum /= H_cum[-1]
                levels_to_plot = np.zeros(len(levels))
                for k, lvl in enumerate(levels):
                    try:
                        levels_to_plot[k] = H_flat[H_cum < lvl][-1]
                    except IndexError:
                        levels_to_plot[k] = H_flat[0]
                levels_to_plot.sort()
                m = np.diff(levels_to_plot) == 0
                if np.any(m):
                    print(f'Warning: levels are the same')
                while np.any(m):
                    levels_to_plot[np.where(m)[0][0]] *= 1.0 - 1e-4
                    m = np.diff(levels_to_plot) == 0
                levels_to_plot.sort()

                # compute the centers of the bins
                X1, Y1 = 0.5*(xedges[1:] + xedges[:-1]), 0.5*(yedges[1:] + yedges[:-1])
                # extend the array to make nicer contours
                H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
                H2[2:-2, 2:-2] = H
                H2[2:-2, 1] = H[:, 0]
                H2[2:-2, -2] = H[:, -1]
                H2[1, 2:-2] = H[0]
                H2[-2, 2:-2] = H[-1]
                H2[1, 1] = H[0, 0]
                H2[1, -2] = H[0, -1]
                H2[-2, 1] = H[-1, 0]
                H2[-2, -2] = H[-1, -1]
                X2 = np.concatenate(
                    [
                        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
                        X1,
                        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
                    ]
                )
                Y2 = np.concatenate(
                    [
                        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
                        Y1,
                        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
                    ]
                )

                # plot the contours
                ax[i, j].contour(X2, Y2, H2.T, levels=levels_to_plot, colors='k', linewidths=0.5)


    # add the data to the corner plot
    if data is not None:
        for i in range(N_dim):
            for j in range(i):
                ax[i, j].axvline(data[j], lw=0.5, c=cdata)
                ax[i, j].axhline(data[i], lw=0.5, c=cdata)
            ax[i, i].axvline(data[i], lw=0.5, c=cdata)


    return fig, ax