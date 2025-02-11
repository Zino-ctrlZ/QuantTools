"""
The purpose of this module is to test the performance of using async vs multiprocessing in the context of the riskmanager.py module by testing 
a known bottleneck in the code, the bottleneck is running data retireval from an api using list of different arguments. 

To run the module appropriately, the following steps should be taken:
run the run_Processes() function, make sure the run_async() function is commented out. The data will be logged to data/ directory .
run the run_async() function, make sure the run_Processes() function is commented out. The data will be logged to data/ directory .

at the end of both runs, you will be able to compare performance by looking at runProcesses.txt and run_async.txt respectively.
"""


import os
from trade.helpers.pools import runProcesses
import asyncio 
import cProfile
import pstats
import io
from dbase.DataAPI.ThetaData import (retrieve_eod_ohlc, retrieve_eod_ohlc_async, retrieve_openInterest, retrieve_openInterest_async)
from trade.helpers.helper import (generate_option_tick_new)

orderedList = [['GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL'],
               ['2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24', '2023-07-24'],
               ['2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21'],
               ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
               ['2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26', '2023-06-26'],
               [132.5, 130.0, 127.5, 125.0, 122.5, 120.0, 117.5, 115.0, 112.5, 110.0, 115.0, 112.5, 110.0, 107.5, 105.0, 102.5, 100.0, 98.0, 97.5, 96.0, 95.0, 94.0, 92.0]]

tickOrderedList = [['GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL'],
                   ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                   ['2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-21'],
                   [132.5, 130.0, 127.5, 125.0, 122.5, 120.0, 117.5, 115.0, 112.5, 110.0, 115.0, 112.5, 110.0, 107.5, 105.0, 102.5, 100.0, 98.0, 97.5, 96.0, 95.0, 94.0, 92.0]]

def test_runProcesses(): 
  #simulate the retrieve_eod_ohlc action in  riskmanager.py
  eod_results = list(runProcesses(retrieve_eod_ohlc, orderedList, 'imap'))
  oi_results = list(runProcesses(retrieve_openInterest, orderedList, 'imap'))
  tick_results = list(runProcesses(generate_option_tick_new, tickOrderedList, 'imap'))
  
  if (tick_results and oi_results and eod_results):
    return (tick_results, oi_results, eod_results)
  

  
async def test_async():
    # Transpose orderedList to generate tuples of arguments
    transposed_ordered_list = list(zip(*orderedList))
    transposed_tick_ordered_list = list(zip(*tickOrderedList))
    # Create tasks for each tuple
    eod_tasks = [asyncio.create_task(retrieve_eod_ohlc_async(*args)) for args in transposed_ordered_list]
    oi_tasks = [asyncio.create_task(retrieve_openInterest_async(*args)) for args in transposed_ordered_list]
    tick_results = [generate_option_tick_new(*args) for args in transposed_tick_ordered_list]
    

    # Run all tasks concurrently
    eod_results, oi_results = await asyncio.gather(
      asyncio.gather(*eod_tasks),
      asyncio.gather(*oi_tasks)
    )
    
    print('async complete')
    return (tick_results, oi_results, eod_results)

  
  
  
if __name__ == '__main__' : 
  profiler = cProfile.Profile()

  # test the runProcesses function, log the results to a file 
  def run_processes():
    try:
      profiler.enable()
      (tick_result, oi_result, eod_result) = test_runProcesses()
      profiler.disable()
        # Save profiling data to a file
      ioStringStream = io.StringIO()
      fstats = pstats.Stats(profiler, stream=ioStringStream).sort_stats('cumulative')
      fstats.print_stats()
      ioStringStream.seek(0)
      data = ioStringStream.read()
      with open(os.path.join(os.path.dirname(__file__), 'data', 'runProcesses.txt'), 'w') as stream:
        print('writing to file')
        stream.write(data)
        stream.flush()
      
      
      with open(os.path.join(os.path.dirname(__file__), 'data','runProcesses_eod_results.csv'), 'w') as f:
        for df in eod_result:
          df.to_csv(f, index=False)
          f.write('\n\n')  # Add spaces between each dataframe
          
      with open(os.path.join(os.path.dirname(__file__), 'data','runProcesses_oi_results.csv'), 'w') as f:
        for df in oi_result:
          df.to_csv(f, index=False)
          f.write('\n\n')  # Add spaces between each dataframe
          
      with open(os.path.join(os.path.dirname(__file__), 'data','tick_results.txt'), 'w') as f:
        for tick in tick_result:
          f.write(str(tick))
          f.write('\n\n')
          
    except Exception as e:
      print('error occured: ', e)
    
  # run_processes()
  
  # test the run_async function, log the results to a file
  def run_async():
    try:
      profiler.enable()
      (tick_result, oi_result, eod_result) = asyncio.run(test_async())
      profiler.disable()
        # Save profiling data to a file
      ioStringStream = io.StringIO()
      fstats = pstats.Stats(profiler, stream=ioStringStream).sort_stats('cumulative')
      fstats.print_stats()
      ioStringStream.seek(0)
      data = ioStringStream.read()
      with open(os.path.join(os.path.dirname(__file__),'data','run_async.txt'), 'w') as stream:
        print('writing to file')
        stream.write(data)
        stream.flush()
      
      with open(os.path.join(os.path.dirname(__file__), 'data','run_async_eod_results.csv'), 'w') as f:
        for df in eod_result:
          df.to_csv(f, index=False)
          f.write('\n\n')  # Add spaces between each dataframe
          
      with open(os.path.join(os.path.dirname(__file__), 'data','run_async_oi_results.csv'), 'w') as f:
        for df in oi_result:
          df.to_csv(f, index=False)
          f.write('\n\n')  # Add spaces between each dataframe
          
      with open(os.path.join(os.path.dirname(__file__), 'data','run_async_tick_results.txt'), 'w') as f:
        for tick in tick_result:
          f.write(str(tick))
          f.write('\n\n')
    
    except Exception as e:
      print('error occured: ', e)
      
  
  run_async()