import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dual_heatmap(lam,labels,k,title):
    grid=np.zeros((2,k,k))
    for val,(z,t,y) in zip(lam,labels):
        z = int(z)
        t = int(t)
        y = int(y)
        grid[t,z,y]=val
    fig,axs=plt.subplots(1,2,figsize=(10,4))

    for t in [0,1]:
        print(grid[t])
        im=axs[t].imshow(grid[t],origin="lower")
        axs[t].set_title(f"T={t}")
        axs[t].set_xlabel("Z")
        axs[t].set_ylabel("Y")
        fig.colorbar(im,ax=axs[t])

    plt.suptitle(title)
    plt.tight_layout()

    savepath = "Plots"
    os.makedirs(savepath, exist_ok=True)

    plt.savefig(os.path.join(savepath, title + ".jpg"), dpi=300)
    plt.close()