from src.optimize.correction_functions import shift, linear
from src.showcase.accuracy_plot import accuracy_plot
from src.showcase.frame_display import show_GCPs_on_frame
from src.utils.RPC_parser import parse_RPCs
from src.optimize.optimize import automated_optimize, optimize_camera_parameters

if __name__ == "__main__":

    # parse RPCs for all supported cities
    parse_RPCs()

    # Run 10 optimizations for PSM model with shift correction function one for every GCP as a training sample in San Francisco
    automated_optimize(model = "PSM", correction_model={"correction_function": shift}, train_set={"San_francisco": [1]})

    # Run one optimization for RFM model with linear correction function on all 10 GCPs in San Francisco
    optimize_camera_parameters(model="RFM", correction_model={"correction_function": linear}, train_GCPs={"San_francisco": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]})

    # Show GCPs on a specific frame in San Francisco using the optimized shift correction function for the PSM model
    show_GCPs_on_frame("1293562080.02321601_sc00113_c1_PAN_i0000000185", optimized_function="shift", city="San_francisco", train_GCPs={"San_francisco": ["1"]}, model="PSM")

    # Show GCPs on a specific frame in San Francisco using the optimized linear correction function for the RFM model
    show_GCPs_on_frame("1293562080.02321601_sc00113_c1_PAN_i0000000185", optimized_function="linear", city="San_francisco", train_GCPs={"San_francisco": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]}, model="RFM")

    # Generate an accuracy plot for the PSM model with shift correction function
    accuracy_plot(no_eval_GCPs=1, optimized_function=shift, model="PSM")