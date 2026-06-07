import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import json

def show(id: str, area: str = "small", ref: bool = False):

    if ref:
        with open("../../landsat/GCPlib_ref.json", "r") as f:
            GCPinfo = json.load(f)
    else:
        with open("../../landsat/GCPlib.json", "r") as f:
            GCPinfo = json.load(f)

    ver = "_01" if area == "small" else "_02"

    path = "../../landsat/image_chips/"+id+ver

    meta_data = GCPinfo[id+ver]

    data = np.fromfile(path, dtype=np.uint8)
    img = data.reshape((64,64))

    x = float(meta_data["ref_sample"])
    y = float(meta_data["ref_line"])

    lat = float(meta_data["lat"])
    lon = float(meta_data["lon"])
    print(f"lat: {lat}, lon: {lon}")
    print(f"x: {x}, y: {y}")

    print(type(x))

    # # Acquire default dots per inch value of matplotlib
    # dpi = matplotlib.rcParams['figure.dpi']
    #
    # # Determine the figures size in inches to fit your image
    # height, width = img.shape
    # figsize = width / float(dpi), height / float(dpi)
    #
    # plt.figure(figsize=figsize)

    plt.imshow(img, cmap="gray", origin="upper")
    # plt.scatter(x, y, c="red", marker=".", s=1)

    plt.show()


if __name__ == "__main__":
    show("0440340225", area="mall")