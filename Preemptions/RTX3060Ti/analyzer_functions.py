def data_loader(noSharedPath=None, sharedPath=None, singlePath=None, preemptIvls: bool=True, single: bool=False, onlyPreemptIvls=False):
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
      if onlyPreemptIvls:
        return single_ivls
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
    if onlyPreemptIvls:
      return no_shared_ivls, shared_ivls
    return noShared, shared, no_shared_ivls, shared_ivls

  return noShared, shared


def single_plot(data, NUM_SAMPLES: int, lowerBound=None, upperBound=None, title=None):
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


def same_plotter(noSharedData, sharedData, NUM_SAMPLES: int, 
                 preemptIvls: bool=True, lowerBound=None, upperBound=None, firstLabel=None, secondLabel=None):
  """Plots the no shared and shared data on the same plot"""
  assert len(sharedData) == len(noSharedData), "Shared and no shared data must be the same length"
  import matplotlib.pyplot as plt
  import numpy as np

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

  # Add x-axis ticks
  plt.xticks(np.linspace(0, 1000000, 11))

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


def plot_separate(noSharedData, sharedData, NUM_SAMPLES: int, 
                  preemptIvls: bool=True, lowerBound=None, upperBound=None, firstLabel=None, secondLabel=None,
                  medianLine=False):
  """Plots the data side-by-side"""
  assert len(sharedData) == len(noSharedData), "Shared and no shared data must be the same length"
  import matplotlib.pyplot as plt
  import numpy as np

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

  # Add x-axis ticks
  ax1.set_xticks(np.linspace(0, 1000000, 11))
  ax2.set_xticks(np.linspace(0, 1000000, 11))

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

  # Plot the interval lines if desired
  if medianLine:
    noSharedMedian = np.median(noSharedData)
    sharedMedian = np.median(sharedData)
    ax1.axhline(noSharedMedian, color='red', linestyle='--', label='Median')
    ax2.axhline(sharedMedian, color='red', linestyle='--', label='Median')

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


def plot_side_by_side(noSharedData, sharedData, NUM_SAMPLES: int, medianOffset=10, blockOffset=10,
                  preemptIvls: bool=True, lowerBound=None, upperBound=None, firstLabel=None, secondLabel=None,
                  medianLines=False, worstCaseLines=False, blockLines=False, medianImpute=False, percent=99, offset=100000, y_axis="Interval (us)", perCap=None,
                  lowerTextOffset=0, upperTextOffset=0, plotOverhead=False):
  """Plots the data side-by-side on the same plot"""
  assert len(sharedData) == len(noSharedData), "Shared and no shared data must be the same length"
  import matplotlib.pyplot as plt
  import numpy as np

  # Create one big plot
  plt.figure(figsize=(15, 7))

  # x-values for no shared data
  if plotOverhead:
    noSharedX = np.arange(1, NUM_SAMPLES + 1)
  else:
    noSharedX = np.arange(1, NUM_SAMPLES)

  # Move the shared data to the right
  sharedX = noSharedX + NUM_SAMPLES + offset

  # Scatterplot of without shared memory
  if firstLabel is not None:
    plt.scatter(noSharedX, noSharedData, label=firstLabel, color='dodgerblue')
  else:
    plt.scatter(noSharedX, noSharedData, label='Without Shared', color='dodgerblue')

  # Add labels and titles
  plt.xlabel('Preemption #')
  plt.ylabel(y_axis)

  # Add y-axis ticks
  plt.locator_params(axis='y', nbins=10)

  # Add x-axis ticks
  xTickLabels = ['']
  tenth = NUM_SAMPLES // 10
  if NUM_SAMPLES == 1000000:
    xTickNums = np.arange(100, 1000, 100)
    xTickLabels += [f'{n}k' for n in xTickNums]
    xTickLabels.append('1 M')
    xTickLabels += ["" for i in range(int(offset / tenth))]
    xTickLabels += [f'{n}k' for n in xTickNums]
    xTickLabels.append('1 M')
    plt.xticks(np.arange(1, 2*NUM_SAMPLES+offset+tenth, tenth), xTickLabels)
  else:
    xTickNums = np.arange(NUM_SAMPLES // 10, NUM_SAMPLES, NUM_SAMPLES // 10)
    xTickLabels += [f'{n}' for n in xTickNums]
    xTickLabels.append(str(NUM_SAMPLES))
    xTickLabels += ["" for i in range(int(offset / tenth))]
    xTickLabels += [f'{n}' for n in xTickNums]
    xTickLabels.append(str(NUM_SAMPLES))
    plt.xticks(np.arange(1, 2*NUM_SAMPLES+offset+tenth, tenth), xTickLabels)

  if preemptIvls:
    if perCap:
      plt.title(f'Preemption and Execution - {perCap}% Capacity')
    elif plotOverhead:
      plt.title('Preemption Overhead')
    else:  
      plt.title('Preemption and Execution')
  else:
    plt.title('Logger Execution Intervals')

  # Set y-axis limits if desired
  if lowerBound is not None and upperBound is not None:
    plt.ylim(lowerBound, upperBound)

  # Scatterplot of with shared memory
  if secondLabel is not None:
    plt.scatter(sharedX, sharedData, label=secondLabel, color='mediumspringgreen')
  else:
    plt.scatter(sharedX, sharedData, label='With Shared', color='mediumspringgreen')

  # Calculate the median and std devs for the lines
  noSharedMedian = np.median(noSharedData)
  sharedMedian = np.median(sharedData)
  noSharedStd = np.std(noSharedData)
  sharedStd = np.std(sharedData)
  if sharedMedian > noSharedMedian:
    lowerMedian = noSharedMedian
    lowerStd = noSharedStd
    upperMedian = sharedMedian
    upperStdDev = sharedStd
  else:
    lowerMedian = sharedMedian
    lowerStd = sharedStd
    upperMedian = noSharedMedian
    upperStd = noSharedStd

  # Set the x coordinate for the interval lines to be in the center of the offset
  intervalLineX = NUM_SAMPLES+offset//2

  # Decide whether the no shared data or shared data is higher for plotting
  if sharedMedian > noSharedMedian:
    sharedHigher=True
  else:
    sharedHigher=False

  # Label the percentile lines for the worst-case and block lines
  lowerLabel, upperLabel = percentile_labels(percent)

  # Plot the interval lines if desired
  if medianLines:

    # Median lines
    plt.plot([0, intervalLineX+offset//5], [noSharedMedian, noSharedMedian], color='black', linestyle='--', label='Median')
    plt.plot([intervalLineX-offset//5, 2*NUM_SAMPLES+offset], [sharedMedian, sharedMedian], color='black', linestyle='--')

    # Calculate the median difference
    medianDifference = upperMedian - lowerMedian

    # Median difference line
    plt.plot([intervalLineX, intervalLineX], [lowerMedian, upperMedian], color='firebrick', linestyle='--', label=f'{medianDifference:.3f} us')

    # Draw the arrows
    plt.annotate('', xy=(intervalLineX, upperMedian), xytext=(intervalLineX-0.001, upperMedian), 
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=0', color='firebrick'))
    plt.annotate('', xy=(intervalLineX, lowerMedian), xytext=(intervalLineX+0.001, lowerMedian),
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=180', color='firebrick'))
                 
    # Put the median difference text
    if lowerTextOffset == 0 or upperTextOffset == 0:
      plt.text(intervalLineX+offset//4, lowerMedian-lowerStd/2, f'{abs(medianDifference):.3f} us', fontsize=12, color='firebrick')
    else:
      plt.text(intervalLineX+upperTextOffset, lowerMedian-lowerTextOffset, f'{abs(medianDifference):.3f} us', fontsize=12, color='firebrick')

    # Calculate the lower and upper bounds based on the provided percentile
    if medianImpute:
      sharedData[sharedData > upperBound] = sharedMedian
      sharedData[sharedData < lowerBound] = sharedMedian
      noSharedData[noSharedData > upperBound] = noSharedMedian
      noSharedData[noSharedData < lowerBound] = noSharedMedian
    sharedLower= np.percentile(sharedData, 100-percent)
    noSharedLower = np.percentile(noSharedData, 100-percent)
    sharedUpper = np.percentile(sharedData, percent)
    noSharedUpper = np.percentile(noSharedData, percent)

    # Put the no shared median above its block
    plt.text(NUM_SAMPLES/2, noSharedUpper+medianOffset, f'{noSharedMedian:.3f} us', fontsize=12, color='black')

    # Put the shared median above its block
    plt.text(NUM_SAMPLES+offset+NUM_SAMPLES/2, sharedUpper+medianOffset, f'{sharedMedian:.3f} us', fontsize=12, color='black')

  elif worstCaseLines:
    # Plot the lines for the top and bottom of one of the upper and lower blocks, respectively
    # Find the bounds based on the provided percentile
    if medianImpute:
      sharedData[sharedData > upperBound] = sharedMedian
      sharedData[sharedData < lowerBound] = sharedMedian
      noSharedData[noSharedData > upperBound] = noSharedMedian
      noSharedData[noSharedData < lowerBound] = noSharedMedian
    if sharedHigher:
      upperBlock = np.percentile(sharedData, percent)
      lowerBlock = np.percentile(noSharedData, 100-percent)
      # Plot the lower bound line
      plt.plot([0, intervalLineX+offset//5], [lowerBlock, lowerBlock], color='black', linestyle='--', label=lowerLabel)

      # Plot the upper bound line
      plt.plot([intervalLineX-offset//5, 2*NUM_SAMPLES+offset], [upperBlock, upperBlock], color='dimgrey', linestyle='--', label=upperLabel)
    else:
      upperBlock = np.percentile(noSharedData, percent)
      lowerBlock = np.percentile(sharedData, 100-percent)
      # Plot the lower bound line
      plt.plot([intervalLineX-offset//5, 2*NUM_SAMPLES+offset], [lowerBlock, lowerBlock], color='dimgrey', linestyle='--', label=lowerLabel)

      # Plot the upper bound line
      plt.plot([0, intervalLineX+offset//5], [upperBlock, upperBlock], color='black', linestyle='--', label=upperLabel)
      
    # Calculate the difference between the upper and lower blocks in the typical worst case
    worstCaseDiff = upperBlock - lowerBlock

    # Line for interval between lower and upper blocks bounds (worst case)
    plt.plot([intervalLineX, intervalLineX], [lowerBlock, upperBlock], color='firebrick', linestyle='--', label=f'{worstCaseDiff:.3f} us')

    # Arrow for upper block
    plt.annotate('', xy=(intervalLineX, upperBlock), xytext=(intervalLineX-0.001, upperBlock), 
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=0', color='firebrick'))
    
    # Arrow for lower block
    plt.annotate('', xy=(intervalLineX, lowerBlock), xytext=(intervalLineX+0.001, lowerBlock),
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=180', color='firebrick'))
    
    # Put the worst case difference text
    plt.text(intervalLineX+offset//4, lowerBlock-blockOffset, f'{worstCaseDiff:.3f} us', fontsize=12, color='firebrick')

  elif blockLines:
    # Plot the lines for the top and bottom of each data block
    # Calculate the lower and upper bounds based on the provided percentile
    if medianImpute:
      sharedData[sharedData > upperBound] = sharedMedian
      sharedData[sharedData < lowerBound] = sharedMedian
      noSharedData[noSharedData > upperBound] = noSharedMedian
      noSharedData[noSharedData < lowerBound] = noSharedMedian
    sharedLower= np.percentile(sharedData, 100-percent)
    noSharedLower = np.percentile(noSharedData, 100-percent)
    sharedUpper = np.percentile(sharedData, percent)
    noSharedUpper = np.percentile(noSharedData, percent)

    # Calculate the differences for each shared block
    sharedDiff = sharedUpper - sharedLower
    noSharedDiff = noSharedUpper - noSharedLower
    
    # Plot the shared lower bound line
    leftOffset = NUM_SAMPLES // 100
    plt.plot([intervalLineX+leftOffset, NUM_SAMPLES+offset], [sharedLower, sharedLower], color='dimgrey', linestyle='--')

    # Plot the shared upper bound line
    plt.plot([intervalLineX+leftOffset, NUM_SAMPLES+offset], [sharedUpper, sharedUpper], color='dimgrey', linestyle='--')

    # Plot the difference line for shared memory
    rightOffset = NUM_SAMPLES // 50
    plt.plot([intervalLineX+rightOffset, intervalLineX+rightOffset], [sharedLower, sharedUpper], color='firebrick', linestyle='--')

    # Plot the no shared lower bound line
    plt.plot([1000000, intervalLineX-10000], [noSharedLower, noSharedLower], color='black', linestyle='--')

    # Plot the no shared upper bound line
    plt.plot([1000000, intervalLineX-10000], [noSharedUpper, noSharedUpper], color='black', linestyle='--')

    # Plot the difference line for no shared memory
    plt.plot([intervalLineX-20000, intervalLineX-20000], [noSharedLower, noSharedUpper], color='firebrick', linestyle='--')

    # Arrow for shared upper bound
    plt.annotate('', xy=(intervalLineX+20000, sharedUpper), xytext=(intervalLineX+20000-0.001, sharedUpper), 
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=0', color='firebrick'))
    
    # Arrow for shared lower bound
    plt.annotate('', xy=(intervalLineX+20000, sharedLower), xytext=(intervalLineX+20000+0.001, sharedLower),
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=180', color='firebrick'))
    
    # Arrow for no shared upper bound
    plt.annotate('', xy=(intervalLineX-20000, noSharedUpper), xytext=(intervalLineX-20000-0.001, noSharedUpper), 
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=0', color='firebrick'))
    
    # Arrow for no shared lower bound
    plt.annotate('', xy=(intervalLineX-20000, noSharedLower), xytext=(intervalLineX-20000+0.001, noSharedLower),
                 arrowprops=dict(arrowstyle='->', linewidth=1.5, connectionstyle='bar,angle=180', color='firebrick'))
    
    # Calculate the offset for text based on the bounds of the plot
    textOffset = (upperBound - lowerBound) * 0.1

    if sharedHigher:
      # Put the shared difference text above the shared data and the no shared below
      plt.text(intervalLineX+blockOffset, sharedUpper+textOffset, f'{sharedDiff:.3f} us', fontsize=12, color='firebrick')
      plt.text(intervalLineX-blockOffset, noSharedLower-textOffset, f'{noSharedDiff:.3f} us', fontsize=12, color='firebrick')
    else:
      # Put the shared difference text below the shared data
      # Put the text for the shared difference and no shared above
      plt.text(intervalLineX+blockOffset, sharedLower-textOffset, f'{sharedDiff:.3f} us', fontsize=12, color='firebrick')
      plt.text(intervalLineX-blockOffset, noSharedUpper+textOffset, f'{noSharedDiff:.3f} us', fontsize=12, color='firebrick')
    
  # Show the plot
  plt.legend(loc='upper right')
  plt.show()


def box_plotter(leftData, rightData):
  """Plots box plots of the data side-by-side"""
  import matplotlib.pyplot as plt

   # Create one big plot
  plt.figure(figsize=(15, 7))

  # Plot both box plots
  plt.boxplot([leftData, rightData])
  plt.show()


# Function to format the percentile label
def percentile_labels(percentile):
  # import numpy as np
  # upperLast = int(str(percentile)[-1])
  # lowerLast = int(str(100-percentile)[-1])

  # # Get upper suffix
  # if upperLast == 1:
  #   upperSuffix = "st"
  # elif upperLast == 2:
  #   upperSuffix = "nd"
  # elif upperLast == 3:
  #   upperSuffix = "rd"
  # else:
  #   upperSuffix = "th"

  # # Get lower suffix
  # if lowerLast == 1:
  #   lowerSuffix = "st"
  # elif lowerLast == 2:
  #   lowerSuffix = "nd"
  # elif lowerLast == 3:
  #   lowerSuffix = "rd"
  # else:
  #   lowerSuffix = "th"
  return f"{100-percentile:.3f}%", f"{percentile}%"


def read_ivls(paths, single=False):
  import numpy as np
  if single:
    data, ivls = data_loader(singlePath=paths, single=True)
    return  np.array(ivls) / 1000
  ivls_array = []
  for path in paths:
    data, ivls = data_loader(singlePath=path, single=True)
    # Convert to microseconds
    ivls = np.array(ivls) / 1000
    ivls_array.append(ivls)
  return ivls_array



def get_overhead(ivls, timeslice_length):
  result = []
  for ivl in ivls:
    result.append((ivl - timeslice_length) / 2)
    result.append((ivl - timeslice_length) / 2)
  return result
 

def cut_ivls(ivls, window):
  """
  Takes an experimment (list of intervals) and only takes the percentage of intervals
  specified by window (e.g. [25, 75] for inter-quartile range)
  """
  n = len(ivls)
  lower = int(window[0] / 100 * n)
  upper = int(window[1] / 100 * n)
  return ivls[lower:upper]