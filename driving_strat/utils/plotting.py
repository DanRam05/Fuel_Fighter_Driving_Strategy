import matplotlib.pyplot as plt

def plot_track(x, y, ax=None, show=True, save_path=None, line_radius_m=None, **kwargs):
    """Plot track coordinates.

    Args:
        x (array-like): x coordinates (e.g., UTM X)
        y (array-like): y coordinates (e.g., UTM Y)
        ax (matplotlib.axes.Axes, optional): axes to draw on.
        show (bool): whether to call `plt.show()`.
        save_path (str or Path, optional): file path to save the figure.
        line_radius_m (float, optional): visual radius (in meters) to represent
            as the plotted line half-width. The plotted line width will be set
            to twice this value. If None, a default linewidth is used.
        **kwargs: forwarded to `ax.plot`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Determine linewidth: if user supplied a radius in meters, use twice that
    # as a visual linewidth. This is a visual mapping, not a strict physical
    # conversion between data units and display points.
    lw = kwargs.pop('lw', None)
    if line_radius_m is not None:
        lw = 2.0 * float(line_radius_m)

    ax.plot(x, y, lw=lw if lw is not None else 1, color=kwargs.pop('color', 'C0'))
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Track')

    if save_path:
        fig.savefig(str(save_path), bbox_inches='tight')

    if show:
        plt.show()

    return ax
