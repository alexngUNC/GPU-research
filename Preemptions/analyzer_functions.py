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