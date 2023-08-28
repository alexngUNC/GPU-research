def data_loader(noSharedPath=None, sharedPath=None, singlePath=None, preemptIvls: bool=True, single: bool=False):
  """Reads the CSV files into Pandas dataframes"""
  import pandas as pd

  # if only one data path is being passed in:
  if single:
    assert singlePath is not None, "Single data path must be passed in"
    single = pd.read_csv(singlePath, header=None)
    NUM_SAMPLES = len(single)
    single.columns = ['start', 'end']

    # calculate logger execution interval
    single['interval'] = single['end'] - single['start']

    # calculate preemption intervals if desired
    if preemptIvls:
      single_ivls = []
      for i in range(0, NUM_SAMPLES-1):
        single_ivls.append(single['start'][i+1] - single['end'][i])
      return single, single_ivls
    return single

  # read in the data
  noShared = pd.read_csv(noSharedPath, header=None)
  shared = pd.read_csv(sharedPath, header=None)

  # make sure that the data has the same number of samples
  NUM_SAMPLES = len(shared)
  assert len(noShared) == NUM_SAMPLES, "Shared and no shared data must have the same length"

  # rename columns
  noShared.columns = ['start', 'end']
  shared.columns = ['start', 'end']

  # calculate logger execution interval
  noShared['interval'] = noShared['end'] - noShared['start']
  shared['interval'] = shared['end'] - shared['start']

  # calculate the preemption intervals if desired
  if preemptIvls:
    no_shared_ivls = []
    shared_ivls = []
    for i in range(0, NUM_SAMPLES-1):
      no_shared_ivls.append(noShared['start'][i+1] - noShared['end'][i])
      shared_ivls.append(shared['start'][i+1] - shared['end'][i])
    return noShared, shared, no_shared_ivls, shared_ivls

  return noShared, shared


def single_plot(data: list[int], NUM_SAMPLES: int, lowerBound=None, upperBound=None, title=None):
  import matplotlib.pyplot as plt
  plt.scatter(range(1, NUM_SAMPLES), data)
  plt.xlabel('Preemption #')
  plt.ylabel('Interval (ns)')
  if title is not None:
    plt.title(title)
  else:
    plt.title('Preemption and Kernel Execution Intervals')

  # Adjust the y-axis limits if desired
  if lowerBound is not None and upperBound is not None:
    plt.ylim(lowerBound, upperBound)
  plt.show()


def same_plotter(noSharedData: list[int], sharedData: list[int], NUM_SAMPLES: int, 
                 preemptIvls: bool=True, lowerBound=None, upperBound=None, firstLabel=None, secondLabel=None):
  """Plots the no shared and shared data on the same plot"""
  assert len(sharedData) == len(noSharedData), "Shared and no shared data must be the same length"
  import matplotlib.pyplot as plt

  # Make a scatterplot of both on the same plot
  if firstLabel is not None and secondLabel is not None:
    plt.scatter(range(1, NUM_SAMPLES), noSharedData, label=firstLabel)
    plt.scatter(range(1, NUM_SAMPLES), sharedData, label=secondLabel)
  else:
    plt.scatter(range(1, NUM_SAMPLES), noSharedData, label='Without shared')
    plt.scatter(range(1, NUM_SAMPLES), sharedData, label='With shared')

  # Add axis labels
  plt.xlabel('Preemption #')
  plt.ylabel('Interval (ns)')

  # Add title based on interval type
  if preemptIvls:
    plt.title('Preemption and Kernel Execution')
  else:
    plt.title('Logger Execution Intervals')

  # Add legend
  plt.legend(loc='upper right')

  # Adjust the y-axis limits if desired
  if lowerBound is not None and upperBound is not None:
    plt.ylim(lowerBound, upperBound)

  # Show the plot
  plt.show()


def plot_separate(noSharedData: list[int], sharedData: list[int], NUM_SAMPLES: int, 
                  preemptIvls: bool=True, lowerBound=None, upperBound=None, firstLabel=None, secondLabel=None):
  """Plots the data side-by-side"""
  assert len(sharedData) == len(noSharedData), "Shared and no shared data must be the same length"
  import matplotlib.pyplot as plt

  # Create the subplots
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

  # Scatterplot of without shared memory
  if firstLabel is not None:
    ax1.scatter(range(1, NUM_SAMPLES), noSharedData, label=firstLabel)
  else:
    ax1.scatter(range(1, NUM_SAMPLES), noSharedData, label='Without Shared')

  # Add labels and titles
  ax1.set_xlabel('Preemption #')
  ax1.set_ylabel('Interval (ns)')
  if preemptIvls:
    ax1.set_title('Preemption and Kernel Execution')
    ax2.set_title('Preemption and Kernel Execution')
  else:
    ax1.set_title('Logger Execution Intervals')
    ax2.set_title('Logger Execution Intervals')
  ax1.legend(loc='upper right')

  # Set y-axis limits if desired
  if lowerBound is not None and upperBound is not None:
    ax1.set_ylim(lowerBound, upperBound)
    ax2.set_ylim(lowerBound, upperBound)

  # Scatterplot of with shared memory
  if secondLabel is not None:
    ax2.scatter(range(1, NUM_SAMPLES), sharedData, label=secondLabel, color='orange')
  else:
    ax2.scatter(range(1, NUM_SAMPLES), sharedData, label='With Shared', color='orange')
  ax2.set_xlabel('Preemption #')
  ax2.set_ylabel('Interval (ns)')
  ax2.legend(loc='upper right')

  # Show the plot
  plt.show()


def five_number_summary(noSharedIvls, sharedIvls, singleIvls=None, single=False):
  import numpy as np

  # If only printing the 5-number summary of one set of intervals
  if single:

    # Get the 5-number summary
    single_sum = np.percentile(singleIvls, [0, 25, 50, 75, 100])

    # Print the 5-number summary
    print('No Shared Memory:\n-----------------')
    print("Minimum:", single_sum[0])
    print("Q1:", single_sum[1])
    print("Median:", single_sum[2])
    print("Q3:", single_sum[3])
    print("Maximum:", single_sum[4])
    print('-----------------\n')

    return single_sum
    
  no_shared_sum = np.percentile(noSharedIvls, [0, 25, 50, 75, 100], method='midpoint')

  # Print the 5-number summaries
  print('No Shared Memory:\n-----------------')
  print("Minimum:", no_shared_sum[0])
  print("Q1:", no_shared_sum[1])
  print("Median:", no_shared_sum[2])
  print("Q3:", no_shared_sum[3])
  print("Maximum:", no_shared_sum[4])
  print('-----------------\n')

  print('Shared Memory:\n-----------------')
  shared_sum = np.percentile(sharedIvls, [0, 25, 50, 75, 100], method='midpoint')
  print("Minimum:", shared_sum[0])
  print("Q1:", shared_sum[1])
  print("Median:", shared_sum[2])
  print("Q3:", shared_sum[3])
  print("Maximum:", shared_sum[4])
  print('-----------------')
  return no_shared_sum, shared_sum


def mean_difference(noSharedIvls, sharedIvls, show=True):
  import numpy as np

  # Get means
  no_shared_mean = np.mean(noSharedIvls)
  shared_mean = np.mean(sharedIvls)

  # Calculate the difference in means
  mean_diff = shared_mean - no_shared_mean

  # Calculate the percent difference
  percent_diff = (mean_diff/no_shared_mean) * 100

  # Print the difference
  if show:
    print(f"No Shared Memory Mean: {no_shared_mean}")
    print(f"Shared Memory Mean: {shared_mean}")
    print("-----------------")
    print(f"Difference in Means: {mean_diff}")
    print(f"Percent Difference: {percent_diff}")

  return mean_diff, percent_diff


def median_difference(noSharedIvls, sharedIvls, show=True):
  import numpy as np

  # Get means
  no_shared_median = np.median(noSharedIvls)
  shared_median = np.median(sharedIvls)

  # Calculate the difference in medians
  median_diff = shared_median - no_shared_median

  # Calculate the percent difference
  percent_diff = (median_diff/no_shared_median) * 100

  # Print the difference
  if show:
    print(f"No Shared Memory Median: {no_shared_median}")
    print(f"Shared Memory Median: {shared_median}")
    print("-----------------")
    print(f"Difference in Medians: {median_diff}")
    print(f"Percent Difference: {percent_diff}")

  return median_diff, percent_diff