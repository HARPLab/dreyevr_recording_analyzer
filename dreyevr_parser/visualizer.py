from typing import Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.use("Agg")


def plot_versus(
    data_x: np.ndarray,
    data_y: np.ndarray,
    name_x: Optional[str] = "X",
    name_y: Optional[str] = "Y",
    units_x: Optional[str] = "",
    units_y: Optional[str] = "",
    trim: Tuple[int, int] = (0, 0),
    lines: Optional[bool] = False,
    colour: Optional[str] = "r",
    dir_path: Optional[str] = "results",
):
    # trim the starts and end of data
    trim_start, trim_end = trim
    max_size = min(len(data_x), len(data_y))
    data_x = data_x[trim_start : max_size - trim_end]
    data_y = data_y[trim_start : max_size - trim_end]

    # create a figure that is 6in x 6in
    fig = plt.figure()

    # the axis limits and grid lines
    plt.grid(True)

    units_str_x = " (" + units_x + ")" if units_x != "" else ""
    units_str_y = " (" + units_y + ")" if units_y != "" else ""
    trim_str = " [" + str(trim_start) + ", " + str(trim_end) + "]"

    # label your graph, axes, and ticks on each axis
    plt.xlabel(name_x + units_str_x, fontsize=16)
    plt.ylabel(name_y + units_str_y, fontsize=16)
    plt.xticks()
    plt.yticks()
    plt.tick_params(labelsize=15)
    if name_x == "":
        plt.title(name_y + trim_str, fontsize=18)
    else:
        plt.title(name_y + " versus " + name_x + trim_str, fontsize=18)

    # plot dots
    if lines:
        plt.plot(data_x, data_y, color=colour, linewidth=1)
    else:
        plt.plot(data_x, data_y, colour + "o")

    # complete the layout, save figure, and show the figure for you to see
    plt.tight_layout()
    # make file and save to disk
    if not os.path.exists(os.path.join(os.getcwd(), dir_path)):
        os.mkdir(dir_path)
    filename = name_y + "_vs_" + name_x + ".png" if name_x != "" else name_y + ".png"
    filename = filename.lower().replace(
        " ", "_"
    )  # all lowercase, use _ instead of spaces
    fig.savefig(os.path.join(dir_path, filename))
    plt.close(fig)
    print(f"output figure to {filename}")


def plot_diff(
    subA,
    subB,
    units="",
    name_A="A",
    name_B="B",
    trim=(0, 0),
    lines=False,
    colour="r",
    dir_path="results",
):
    # trim the starts and end of data
    trim_start, trim_end = trim
    max_size = min(len(subA), len(subB))
    subA = subA[trim_start : max_size - trim_end]
    subB = subB[trim_start : max_size - trim_end]

    # create a figure that is 6in x 6in
    fig = plt.figure()

    # the axis limits and grid lines
    plt.grid(True)

    units_str = " (" + units + ")" if units != "" else ""
    trim_str = " [" + str(trim_start) + ", " + str(trim_end) + "]"

    # label your graph, axes, and ticks on each axis
    plt.xlabel("Points", fontsize=16)
    plt.ylabel("Difference" + units_str, fontsize=16)
    plt.xticks()
    plt.yticks()
    plt.tick_params(labelsize=15)
    plt.title("Difference (" + name_A + " - " + name_B + ")" + trim_str, fontsize=18)

    # generate data
    x_data = np.arange(len(subA))
    y_data = subA - subB
    plt.plot(x_data, y_data, colour + "o")
    if lines:
        plt.plot(x_data, y_data, color=colour, linewidth=1)

    # complete the layout, save figure, and show the figure for you to see
    plt.tight_layout()

    # make file and save to disk
    if not os.path.exists(os.path.join(os.getcwd(), dir_path)):
        os.mkdir(dir_path)
    filename = name_A + "_minus_" + name_B + ".png"
    fig.savefig(os.path.join(dir_path, filename))
    plt.close(fig)
