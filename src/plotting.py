import matplotlib.pyplot as plt

def plot_results(df, species_to_plot):
    plt.figure(figsize=(10, 6))
    for species in species_to_plot:
        if species in df.columns:
            plt.plot(df['time'], df[species], label=species)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Species Concentration vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("concentration_plot.png")
    print("Plot saved to concentration_plot.png")
