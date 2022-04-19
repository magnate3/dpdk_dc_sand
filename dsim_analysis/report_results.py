# Display results from tests
import matplotlib.pyplot as plt

def display_wgn_results(results):
    for entry in results:
        print(f'Pol0 - Scale: {entry[0]}   Mean: {entry[1][0]}    StdDev: {entry[2][0]}    Var: {entry[3][0]}')
        print(f'Pol1 - Scale: {entry[0]}   Mean: {entry[1][1]}    StdDev: {entry[2][1]}    Var: {entry[3][1]}')

    plot_hist(results[0][4])
    plot_allan_var(results[0][5])

def display_cw_results(results):
    plot_hist(results[0])


def plot_hist(hist_results):
    plt.figure(1)
    plt.plot(hist_results[0][0])
    plt.plot(hist_results[1][0])
    plt.title('Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.legend(['Pol0', 'Pol1'])
    plt.show()

def plot_allan_var(allan_var_results):
    t2_p0,ad_p0 = allan_var_results[0]
    t2_p1,ad_p1 = allan_var_results[1] 
    plt.figure(2)
    plt.plot(t2_p0,ad_p0)
    plt.plot(t2_p1,ad_p1)
    plt.title('Allan Variance')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Time Cluster')
    plt.ylabel('Allan Deviation')
    plt.legend(['Pol0', 'Pol1'])
    plt.show()