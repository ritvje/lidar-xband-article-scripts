"""Draw scatterplots of lidar and X band radar data."""
import os
import argparse
import warnings
from datetime import datetime
import numpy as np
from numpy.random import default_rng
from scipy import stats
import ennemi as mi
import textwrap
import pandas as pd
import zarr
from pathlib import Path
import matplotlib as mlt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse, bias
import config as cfg
from radar_plotting.plotting import draft, final, use_tex
import utils

warnings.simplefilter(action="ignore")


def plot_linear_fit_plot(
    fig, df, title=None, xlabel=None, ylabel=None, outfn=None, alpha=0.05
):
    """Plot a scatter plot with linear fit line.

    Plots a figure with a scatterplot of the data along with a linear fit.pdf
    Plots also residuals in a second panel.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure that the data is plotted to.
    df : pandas.DataFrame
        Dataframe with the gridded measurements.
    title : str
        Plot title.
    xlabel : str
        x-axis label. First word assumed to be instrument name.
    ylabel : str
        y-axis label. First word assumed to be instrument name.
    outfn : str
        Output file path.
    alpha : float
        Significance level for prediction interval.

    """
    # Calculate linear fit
    X = sm.add_constant(df["Lidar Doppler velocity [m s$^{-1}$]"].values)
    y = df["X-band Doppler velocity [m s$^{-1}$]"].values

    model = sm.OLS(y, X)
    results = model.fit()
    df["Residual [m s$^{-1}$]"] = results.resid
    frame = results.get_prediction(X).summary_frame(alpha=alpha)
    df["Pred.X-band"] = frame["mean"].values

    RMSE = rmse(y, df["Lidar Doppler velocity [m s$^{-1}$]"].values)
    BIAS = bias(y, df["Lidar Doppler velocity [m s$^{-1}$]"].values)

    # Write statistics to text file
    with open(outfn.as_posix().split(".")[0] + "_stats.txt", "w") as statfile:
        statfile.write(results.summary2().as_text())
        statfile.write(f"\nRMSE = {RMSE}")
        statfile.write(f"\nBIAS = {BIAS}")

    #################################################################################
    # Scatterplot figure
    gs = gridspec.GridSpec(
        1,
        1,  # width_ratios=[1, 0.5, 0.5, 0.5], height_ratios=[1],
        figure=fig,
    )
    ax = fig.add_subplot(gs[0, 0])
    minmax_vel = 50

    std_max = np.round(np.sqrt(df["Doppler V var [m$^2$s$^{-2}$]"].values).max(), 1)

    cmap = "viridis_r"
    norm = mlt.colors.Normalize(vmin=0, vmax=std_max)
    sns.scatterplot(
        x=df["Lidar Doppler velocity [m s$^{-1}$]"].values,
        y=df["X-band Doppler velocity [m s$^{-1}$]"].values,
        ax=ax,
        zorder=100,
        # hue=df["Doppler V var [m$^2$s$^{-2}$]"].values,
        color="k",
        linewidth=0,
        alpha=0.05,
        s=1,
        hue_norm=norm,
        palette=cmap,
        legend=None,
    )

    label = f"{ylabel.split(' ')[0]} = {results.params[1]:.2f} * {xlabel.split(' ')[0]}"
    if results.pvalues[0] < 0.05:
        label += f" + {results.params[0]:.2f}"

    ax.plot(
        (-minmax_vel, minmax_vel),
        (results.predict([1, -minmax_vel]), results.predict([1, minmax_vel])),
        ls="-",
        color="tab:orange",
        zorder=110,
        label=label,
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.suptitle(title)

    ax.grid()
    ax.set_aspect(1)
    ax.set_xlim([-minmax_vel, minmax_vel])
    ax.set_ylim([-minmax_vel, minmax_vel])
    ax.set_aspect(1)
    ax.plot(
        (-minmax_vel, minmax_vel),
        (-minmax_vel, minmax_vel),
        ls=":",
        color="k",
        zorder=105,
    )
    fit_legend = ax.legend(loc=1)
    fit_legend.set_zorder(110)

    #################################################################################
    # Correlation figure
    # Mutual information
    mi_pairwise = mi.pairwise_mi(
        df[["Residual [m s$^{-1}$]", *inst_columns[2:]]], normalize=True, drop_nan=True
    )
    np.fill_diagonal(mi_pairwise.values, 1.0)

    # Pearson correlation
    pcorr = np.corrcoef(df[["Residual [m s$^{-1}$]", *inst_columns[2:]]], rowvar=False)
    # np.fill_diagonal(pcorr, np.nan)
    mask = np.zeros_like(pcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Create figure with axis
    corr_fig, (pc_ax, mi_ax) = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)
    divider = make_axes_locatable(mi_ax)
    mi_cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    divider = make_axes_locatable(pc_ax)
    pc_cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)

    mi_norm = mlt.colors.Normalize(vmin=0.0, vmax=0.8)
    mi_cmap = "rocket"
    sns.heatmap(
        mi_pairwise,
        mask=mask,
        ax=mi_ax,
        fmt=".2f",
        yticklabels=False,
        xticklabels=False,
        norm=mi_norm,
        cmap=mi_cmap,
        cbar=False,
        annot=True,
        linewidths=0.1,
        annot_kws=dict(fontsize=4),
    )
    mi_cbar = fig.colorbar(
        mlt.cm.ScalarMappable(norm=mi_norm, cmap=mi_cmap), cax=mi_cax, extend="both"
    )
    mi_cbar.ax.tick_params(labelsize=4)
    # mi_cbar.set_label(label="Correlation coefficient", fontsize=4, weight='bold')

    pc_norm = mlt.colors.Normalize(vmin=-0.8, vmax=0.8)
    pc_cmap = "icefire"
    sns.heatmap(
        pcorr,
        mask=mask,
        ax=pc_ax,
        fmt=".2f",
        yticklabels=False,
        xticklabels=False,
        norm=pc_norm,
        cmap=pc_cmap,
        cbar=False,
        annot=True,
        linewidths=0.1,
        annot_kws=dict(fontsize=4),
    )
    pc_cbar = fig.colorbar(
        mlt.cm.ScalarMappable(norm=pc_norm, cmap=pc_cmap), cax=pc_cax, extend="both"
    )
    pc_cbar.ax.tick_params(labelsize=4)
    # pc_cbar.set_label(label="Pearson correlation coefficient",
    #                   fontsize=4, weight='bold')

    names = [
        "Residual",
        "X-band velocity",
        "Lidar velocity",
        "Lidar CNR",
        "X-band Doppler velocity variance",
        "X-band SNR",
        "X-band reflectivity",
    ]
    names = [textwrap.fill(s, 20) for s in names]
    for ax in [pc_ax, mi_ax]:
        ax.set_xticks(np.arange(len(names)) + 0.5)
        ax.set_yticks(np.arange(len(names)) + 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels(names, fontsize=4)
        # plt.setp(mi_ax.get_xticklabels(), backgroundcolor="white")
        ax.set_aspect(1)

    pc_ax.set_title(
        "(a) Pearson correlation coefficient",
        bbox=dict(facecolor="white", edgecolor="none"),
        fontsize="x-small",
        y=-0.17,
    )
    mi_ax.set_title(
        "(b) Mutual information correlation coefficient",
        bbox=dict(facecolor="white", edgecolor="none"),
        fontsize="x-small",
        y=-0.17,
    )
    corr_fig.subplots_adjust(wspace=0.5)
    corr_fig.suptitle(title, y=0.96)
    corr_fig.savefig(
        outfn.as_posix().split(f".{SCATTERPLOT_EXT}")[0] + "_corr.png",
        dpi=500,
        bbox_inches="tight",
    )

    fig.subplots_adjust(hspace=0.15)
    fig.savefig(outfn, bbox_inches="tight")


def filter_data(
    data,
    xband_ind=0,
    lidar_ind=1,
    lidar_cnr_ind=2,
    xband_var_ind=3,
    xband_snr_ind=4,
    xband_dbz_ind=5,
):
    # Remove very low lidar values
    # data[np.abs(data[:, lidar_ind]) < 0.3, ...] = np.nan

    # Some impossibly small values in Doppler VRAD variance, probably due to gridding
    data[np.abs(data[:, xband_var_ind]) < 0.008, ...] = np.nan

    # data[np.abs(data[:, lidar_cnr_ind]) < -20.0, ...] = np.nan

    # Remove rows with any nan
    data = data[~np.any(np.isnan(data), axis=1), ...]
    return data


def plot_pairgrid(df, fname, title=None):
    xvars = (  # variable name, xbins, xtick multiple
        # ("X-band Doppler velocity [m s$^{-1}$]", np.linspace(-30, 30, 240), 5.),
        # ("Lidar Doppler velocity [m s$^{-1}$]", np.linspace(-30, 30, 240), 5.,),
        # ("CNR [dB]", np.linspace(-30, 10, 240), 5.,),
        # ("Doppler V var [m$^2$s$^{-2}$]", np.linspace(0, 1.0, 100), 0.1,),
        # ("Reflectivity [dBZ]", np.linspace(-25, 50, 240), 5.,),
        # ("SNR [dB]", np.linspace(-25, 50, 240), 5.,),
        (
            "TA_PT1M_AVG",
            np.arange(-5, 30, 1.0),
            5.0,
        ),
        (
            "TD_PT1M_AVG",
            np.arange(-5, 20, 1.0),
            5.0,
        ),
        (
            "PA_PT1M_AVG",
            np.arange(995, 1030, 1.0),
            5.0,
        ),
        (
            "WS_PT10M_AVG",
            np.arange(0, 25, 1.0),
            5.0,
        ),
        (
            "WS_PT2M_AVG",
            np.arange(0, 25, 1.0),
            5.0,
        ),
        (
            "WD_PT10M_AVG",
            np.arange(0, 360, 10),
            30,
        ),
        (
            "WG_PT10M_MAX",
            np.arange(0, 25, 1.0),
            5.0,
        ),
        (
            "WG_PT2M_MAX",
            np.arange(0, 25, 1.0),
            5.0,
        ),
        (
            "CLHB_PT1M_INSTANT",
            np.arange(0, 10e3, 1e2),
            10e2,
        ),
        (
            "VIS_PT1M_AVG",
            np.arange(0, 70e3, 5e2),
            10e3,
        ),
        (
            "CLA_PT1M_ACC",
            np.arange(0, 11, 1.0),
            1.0,
        ),
        (
            "CLA1_PT1M_ACC",
            np.arange(0, 11, 1.0),
            1.0,
        ),
    )
    yvar = "Residual [m s$^{-1}$]"
    nrows = 3
    ncols = int(np.ceil(len(xvars) / nrows))
    fig, axes = plt.subplots(
        figsize=(3 * ncols, 3 * nrows),
        nrows=nrows,
        ncols=ncols,
        squeeze=True,
        sharey=True,
    )

    high_norm = mlt.colors.LogNorm(vmin=1e-3, vmax=0.1)
    low_norm = mlt.colors.LogNorm(vmin=1e-8, vmax=1e-2)
    high_cmap = "flare_r"
    low_cmap = "crest_r"
    for ax, (xvar, xbin, xtick) in zip(axes.flat, xvars):
        ax.hist2d(
            x=df[xvar].values,
            y=df[yvar].values,
            bins=[xbin, np.linspace(-15, 15, 200)],
            zorder=100,
            norm=(high_norm if xbin[-1] < 100 else low_norm),
            cmap=(high_cmap if xbin[-1] < 100 else low_cmap),
            density=True,
        )
        ax.set_xlabel(xvar)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
        ax.grid()
        ax.tick_params(labelsize=6)

    for ax in axes[:, 0]:
        ax.set_ylabel(yvar)

    # divider = make_axes_locatable(resid_ax2)
    # cax2 = divider.append_axes("top", size="3%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(
        mlt.cm.ScalarMappable(cmap=high_cmap, norm=high_norm),
        ax=axes[:, : ncols // 2],
        # ax=axes,
        label="Density",
        orientation="horizontal",
        extend="both",
        shrink=0.35,
    )
    fig.colorbar(
        mlt.cm.ScalarMappable(cmap=low_cmap, norm=low_norm),
        ax=axes[:, ncols // 2 :],
        label="Density",
        orientation="horizontal",
        extend="both",
        shrink=0.35,
    )
    # cbar.ax.xaxis.set_label_position("top")
    # cbar.ax.xaxis.set_ticks_position("top")
    fig.suptitle(title)
    fig.savefig(fname, dpi=500, bbox_inches="tight")


def plot_lidar_pairgrid(df, fname, title=None):
    xvars = (  # variable name, xbins, xtick multiple
        ("X-band Doppler velocity [m s$^{-1}$]", np.linspace(-50, 50, 300), 5.0),
        # ("Lidar Doppler velocity [m s$^{-1}$]", np.linspace(-30, 30, 240), 5.,),
        (
            "CNR [dB]",
            np.linspace(-50, 10, 300),
            5.0,
        ),
        (
            "Doppler V var [m$^2$s$^{-2}$]",
            np.linspace(0, 1.0, 100),
            0.1,
        ),
        (
            "Reflectivity [dBZ]",
            np.linspace(-25, 50, 240),
            5.0,
        ),
        (
            "SNR [dB]",
            np.linspace(-25, 50, 240),
            5.0,
        ),
        # ("TA_PT1M_AVG", np.arange(-5, 30, 1.0), 5.,),
        # ("TD_PT1M_AVG", np.arange(-5, 20, 1.0), 5.,),
        # ("PA_PT1M_AVG", np.arange(995, 1030, 1.0), 5., ),
        # ("WS_PT10M_AVG", np.arange(0, 25, 1.0), 5.,),
        # ("WS_PT2M_AVG", np.arange(0, 25, 1.0), 5.,),
        # ("WD_PT10M_AVG", np.arange(0, 360, 10), 30,),
        # ("WG_PT10M_MAX", np.arange(0, 25, 1.0), 5.,),
        # ("WG_PT2M_MAX", np.arange(0, 25, 1.0), 5.,),
        # ("CLHB_PT1M_INSTANT", np.arange(0, 10e3, 1e2), 10e2,),
        # ("VIS_PT1M_AVG", np.arange(0, 70e3, 5e2), 10e3,),
        # ("CLA_PT1M_ACC", np.arange(0, 11, 1.0), 1.,),
        # ("CLA1_PT1M_ACC", np.arange(0, 11, 1.0), 1.,),
    )
    yvar = "Lidar Doppler velocity [m s$^{-1}$]"
    nrows = 1
    ncols = int(np.ceil(len(xvars) / nrows))
    fig, axes = plt.subplots(
        figsize=(2.5 * ncols, 3.5 * nrows),
        nrows=nrows,
        ncols=ncols,
        squeeze=True,
        sharey=True,
    )

    high_norm = mlt.colors.LogNorm(vmin=1e-3, vmax=0.1)
    low_norm = mlt.colors.LogNorm(vmin=1e-8, vmax=1e-2)
    high_cmap = "flare_r"
    low_cmap = "crest_r"
    for ax, (xvar, xbin, xtick) in zip(axes.flat, xvars):
        ax.hist2d(
            x=df[xvar].values,
            y=df[yvar].values,
            bins=[xbin, np.linspace(-50, 50, 300)],
            zorder=100,
            norm=(high_norm if xbin[-1] < 100 else low_norm),
            cmap=(high_cmap if xbin[-1] < 100 else low_cmap),
            density=True,
        )
        ax.set_xlabel(xvar)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
        ax.grid()
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel(yvar)

    # divider = make_axes_locatable(resid_ax2)
    # cax2 = divider.append_axes("top", size="3%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(
        mlt.cm.ScalarMappable(cmap=high_cmap, norm=high_norm),
        ax=axes,
        label="Density",
        orientation="horizontal",
        extend="both",
        shrink=0.35,
    )
    # fig.colorbar(
    #     mlt.cm.ScalarMappable(
    #         cmap=low_cmap, norm=low_norm),
    #     ax=axes[ncols // 2 - 1:],
    #     label="Density",
    #     orientation="horizontal",
    #     extend="both", shrink=0.35)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.xaxis.set_ticks_position('top')
    fig.suptitle(title)
    fig.savefig(fname, dpi=500, bbox_inches="tight")


def plot_radar_pairgrid(df, fname, title=None):
    xvars = (  # variable name, xbins, xtick multiple
        # ("X-band Doppler velocity [m s$^{-1}$]", np.linspace(-30, 30, 240), 5.),
        (
            "Lidar Doppler velocity [m s$^{-1}$]",
            np.linspace(-50, 50, 300),
            5.0,
        ),
        (
            "CNR [dB]",
            np.linspace(-50, 10, 300),
            5.0,
        ),
        (
            "Doppler V var [m$^2$s$^{-2}$]",
            np.linspace(0, 1.0, 100),
            0.1,
        ),
        (
            "Reflectivity [dBZ]",
            np.linspace(-25, 50, 240),
            5.0,
        ),
        (
            "SNR [dB]",
            np.linspace(-25, 50, 240),
            5.0,
        ),
        # ("TA_PT1M_AVG", np.arange(-5, 30, 1.0), 5.,),
        # ("TD_PT1M_AVG", np.arange(-5, 20, 1.0), 5.,),
        # ("PA_PT1M_AVG", np.arange(995, 1030, 1.0), 5., ),
        # ("WS_PT10M_AVG", np.arange(0, 25, 1.0), 5.,),
        # ("WS_PT2M_AVG", np.arange(0, 25, 1.0), 5.,),
        # ("WD_PT10M_AVG", np.arange(0, 360, 10), 30,),
        # ("WG_PT10M_MAX", np.arange(0, 25, 1.0), 5.,),
        # ("WG_PT2M_MAX", np.arange(0, 25, 1.0), 5.,),
        # ("CLHB_PT1M_INSTANT", np.arange(0, 10e3, 1e2), 10e2,),
        # ("VIS_PT1M_AVG", np.arange(0, 70e3, 5e2), 10e3,),
        # ("CLA_PT1M_ACC", np.arange(0, 11, 1.0), 1.,),
        # ("CLA1_PT1M_ACC", np.arange(0, 11, 1.0), 1.,),
    )
    yvar = "X-band Doppler velocity [m s$^{-1}$]"
    nrows = 1
    ncols = int(np.ceil(len(xvars) / nrows))
    fig, axes = plt.subplots(
        figsize=(2.5 * ncols, 3.5 * nrows),
        nrows=nrows,
        ncols=ncols,
        squeeze=True,
        sharey=True,
    )

    high_norm = mlt.colors.LogNorm(vmin=1e-3, vmax=0.1)
    low_norm = mlt.colors.LogNorm(vmin=1e-8, vmax=1e-2)
    high_cmap = "flare_r"
    low_cmap = "crest_r"
    for ax, (xvar, xbin, xtick) in zip(axes.flat, xvars):
        ax.hist2d(
            x=df[xvar].values,
            y=df[yvar].values,
            bins=[xbin, np.linspace(-30, 30, 240)],
            zorder=100,
            norm=(high_norm if xbin[-1] < 100 else low_norm),
            cmap=(high_cmap if xbin[-1] < 100 else low_cmap),
            density=True,
        )
        ax.set_xlabel(xvar)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
        ax.grid()
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel(yvar)

    # divider = make_axes_locatable(resid_ax2)
    # cax2 = divider.append_axes("top", size="3%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(
        mlt.cm.ScalarMappable(cmap=high_cmap, norm=high_norm),
        ax=axes,
        label="Density",
        orientation="horizontal",
        extend="both",
        shrink=0.35,
    )
    # fig.colorbar(
    #     mlt.cm.ScalarMappable(
    #         cmap=low_cmap, norm=low_norm),
    #     ax=axes[ncols // 2 - 1:],
    #     label="Density",
    #     orientation="horizontal",
    #     extend="both", shrink=0.35)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.xaxis.set_ticks_position('top')
    fig.suptitle(title)
    fig.savefig(fname, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "rtype",
        type=str,
        help="X-band type",
        choices=["WND-01", "WND-02", "WND-03", "MWS-PPI1_G"],
    )
    argparser.add_argument(
        "datapath", type=str, help="Path where datafiles are located"
    )
    argparser.add_argument("startdate", type=str, help="the startdate (YYYYmm)")
    argparser.add_argument("enddate", type=str, help="the enddate (YYYYmm)")
    argparser.add_argument("--outpath", type=str, default=".", help="Output path")

    args = argparser.parse_args()
    startdate = datetime.strptime(args.startdate, "%Y%m")
    enddate = datetime.strptime(args.enddate, "%Y%m")
    plt.style.use(cfg.STYLE_FILE)

    SCATTERPLOT_EXT = "png"

    outpath = Path(args.outpath)

    date_suffix = f"{startdate:%Y%m}_{enddate:%Y%m}"

    rtype = args.rtype
    datapath = args.datapath

    title = f"X-band {rtype} vs. lidar"
    xlabel = "Lidar Doppler velocity [m s$^{-1}$]"
    ylabel = "X-band Doppler velocity [m s$^{-1}$]"
    pair = "xl"
    outfn = (
        outpath / f"{rtype}_{pair}_scatterplot_all_data_{date_suffix}.{SCATTERPLOT_EXT}"
    )

    # Load data as numpy arrays
    xband_ind = 2
    lidar_ind = 3
    lidar_cnr_ind = 4
    xband_var_ind = 5
    xband_snr_ind = 6
    xband_dbz_ind = 7
    data = None
    dateinterval = pd.date_range(startdate, enddate + pd.offsets.MonthEnd(), freq="M")
    for month in dateinterval:
        fn = os.path.join(
            datapath, f"scatterplot_{month:%Y%m}_{month:%Y%m}_{rtype}_{pair}"
        )
        try:
            if data is None:
                data = zarr.load(fn)
            else:
                arr = zarr.load(fn)
                if arr is not None:
                    data = np.concatenate([data, arr], axis=0)
                    del arr
        except Exception as e:
            raise e

    # Clean up data
    data = filter_data(
        data,
        xband_ind=xband_ind,
        lidar_ind=lidar_ind,
        lidar_cnr_ind=lidar_cnr_ind,
        xband_var_ind=xband_var_ind,
        xband_snr_ind=xband_snr_ind,
        xband_dbz_ind=xband_dbz_ind,
    )

    # Get as dataframe
    inst_columns = [
        "X-band time [UTC]",
        "Lidar time [UTC]",
        "X-band Doppler velocity [m s$^{-1}$]",
        "Lidar Doppler velocity [m s$^{-1}$]",
        "CNR [dB]",
        "Doppler V var [m$^2$s$^{-2}$]",
        "SNR [dB]",
        "Reflectivity [dBZ]",
    ]
    df = pd.DataFrame(
        data,
        columns=inst_columns,
    )
    df["X-band time [UTC]"] = pd.to_datetime(df["X-band time [UTC]"], unit="s")
    df["Lidar time [UTC]"] = pd.to_datetime(df["Lidar time [UTC]"], unit="s")
    df.set_index("Lidar time [UTC]", drop=False, inplace=True)
    df.index.rename("time", inplace=True)
    df.sort_index(axis=0, inplace=True)

    # Delete data so we dont take up memory
    del data

    mintime = df["Lidar time [UTC]"].min().to_pydatetime()
    maxtime = df["Lidar time [UTC]"].max().to_pydatetime()

    # Get weather data
    fmisid = 100968
    params = [
        "stationname",
        "TA_PT1M_AVG",
        "TD_PT1M_AVG",
        "WS_PT10M_AVG",
        "WS_PT2M_AVG",
        "WD_PT10M_AVG",
        "WG_PT10M_MAX",
        "sunelevation",
        "CLHB_PT1M_INSTANT",
        "VIS_PT1M_AVG",
        "CLA1_PT1M_ACC",
        "CLA_PT1M_ACC",
        "WG_PT2M_MAX",
        "PA_PT1M_AVG",
    ]
    dff = df
    # wdf = utils.query_Smartmet_station(fmisid, mintime, maxtime, params)
    # wdf.set_index("time", inplace=True)
    # wdf.sort_index(axis=0, inplace=True)

    # # Merge the dataframes
    # tol = pd.Timedelta("1 minute")
    # # First merge in one direction
    # left_merge = pd.merge_asof(
    #     df,
    #     wdf,
    #     right_index=True,
    #     left_index=True,
    #     direction="nearest",
    #     tolerance=tol,
    # )
    # # Then other
    # right_merge = pd.merge_asof(
    #     wdf,
    #     df,
    #     right_index=True,
    #     left_index=True,
    #     direction="nearest",
    #     tolerance=tol,
    # )
    # # Finally merge the two intermediate dataframes
    # dff = (
    #     left_merge.merge(right_merge, how="outer")
    #     .sort_values(["X-band time [UTC]", "Lidar time [UTC]"])
    #     .reset_index(drop=True)
    # )
    # del right_merge, left_merge, wdf, df
    # # Drop possibly created rows with no remote sensing data
    # dff.dropna(axis=0, how="all", subset=inst_columns, inplace=True)

    # Plot data
    final(
        figsize=(7, 6),
        dpi=500,
        # constrained_layout=True
    )(plot_linear_fit_plot)(
        df,
        xlabel=xlabel,
        ylabel=ylabel,
        outfn=outfn,
    )

    fname = outpath / f"{rtype}_{pair}_pairplot_lidar_data_{date_suffix}.png"
    plot_lidar_pairgrid(df, fname, title=title)

    fname = outpath / f"{rtype}_{pair}_pairplot_radar_data_{date_suffix}.png"
    plot_radar_pairgrid(df, fname, title=title)
