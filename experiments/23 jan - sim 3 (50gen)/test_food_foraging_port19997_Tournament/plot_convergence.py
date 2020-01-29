import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd
from ast import literal_eval

sns.set_style("darkgrid")
sns.set_context("talk")

with open("results.txt", "r") as f:
	fit = f.readlines()

plot_fit = []
for x in fit:
	x = x.replace("\n","").split(" ")
	if not x == " ":
		plot_fit.append(x)

df = pd.DataFrame(plot_fit, columns=["gen","fit"])[1:].astype(float)

#df.groupby('gen', as_index=False)['fit'].mean()
plt_data= []
for x in set(df["gen"]):
	plt_data.append([x, df[df["gen"] == x]["fit"].astype(float).mean()])

plt_data = pd.DataFrame(plt_data, columns=["gen", "fit"]).astype(float)
sns.lineplot(x="gen", y = "fit", data=df)
plt.title("Convergence")
plt.show()
# plt.savefig("plot_sns.png", dpi=300)

"""
with open("positions.txt", "r") as f:
	dat = f.read()
dat = dat.split("\n")
dat = [x for x in dat if x]

#fit = fit.split("\n")

popsize = len([x for x in os.listdir("gen_0") if "all_data" not in x])
n_gens = len([x for x in os.listdir(".") if "gen_" in x])

plot_data = pd.DataFrame(columns=["gen", "x", "y", "z"])
gen = 0
x = []
y = []
for i, individual in enumerate(dat):
#	gen = (i + 1 % popsize) - 1
	gen = individual[0]
	print(gen)
	pos = [literal_eval(x_) for x_ in individual.replace(str(gen)+" [array(", "").replace(")]", "").replace(", array","").split(")(")]
	for p in pos:
		x.append(p[0])
		y.append(p[1])

#ax = sns.kdeplot(x, y, cmap="Blues", shade=True, shade_lowest=False)
#ax.set_frame_on(False)
#plt.xlim(-1, 5)
#plt.ylim(-1, 3)
#plt.axis("off")
#fig = ax.get_figure()
#plt.show()
#fig.savefig("kde.png", bbox_inches="tight", pad_inches=0)
## Arena 1
# up left:   x = 4.23; y = 1.9; z = 0.4
# up right:  x =-2   ; y = 1.9; z = 0.4
# down left: x =-4.23; y =-0.3; z = 0.4
# down right:x =-2   ; y =-0.3; z = 0.4

## Arena 2
# up left   = up right Arena 1
# down left = down right Arena 1
# up right:  x = 0.23; y = 1.9; z = 0.4
# down right:x = 0.23; y =-0.3; z = 0.4
"""
