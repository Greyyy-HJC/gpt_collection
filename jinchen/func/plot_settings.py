"""
Here are the settings for plots in liblattice.
"""

#! color settings
grey = "#808080"
red = "#FF6F6F"
peach = "#FF9E6F"
orange = "#FFBC6F"
sunkist = "#FFDF6F"
yellow = "#FFEE6F"
lime = "#CBF169"
green = "#5CD25C"
turquoise = "#4AAB89"
blue = "#508EAD"
grape = "#635BB1"
violet = "#7C5AB8"
fuschia = "#C3559F"

color_ls = [
    blue,
    orange,
    green,
    red,
    violet,
    fuschia,
    turquoise,
    grape,
    lime,
    peach,
    sunkist,
    yellow,
]


#! marker settings
marker_ls = [
    ".",
    "o",
    "*",
    "P",
    "X",
    "s",
    "p",
    "D",
    "<",
    ">",
    "^",
    "v",
    "1",
    "2",
    "3",
    "4",
    "+",
    "x",
]


#! font settings
font_config = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    # "font.serif": ["Times New Roman"],
}
from matplotlib import rcParams

rcParams.update(font_config)


#! figure size settings
fig_width = 6.75  # in inches, 2x as wide as APS column
gr = 1.618034333  # golden ratio
fig_size = (fig_width, fig_width / gr)


#! errorbar plot settings
errorb = {
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1,
}  # circle
