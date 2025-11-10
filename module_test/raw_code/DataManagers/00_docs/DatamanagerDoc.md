## DATA MANAGERS

Data Manager is an efficient way to handle data retrieved from various api. It will handle all processing, retrieval and storing of data to MySQL Database.

**Main Classes:**
- OptionDataManager: Over-arching manager responsible for returning & storing all option related data'
- BulkDataManager: Over-arching manager responsible for returning Bulk requests.

**Children Classes:**
Options produce different datasets. To ensure readability and reduce errors, we will be seperating each data we expect into child classes
- SpotDataManager
- VolDataManager
- GreeksDataManager
- AtrributionDataManager
- ChainDataManager
- ScenarioDataManager

**Misc Classes:**
Any other class that will be used to streamline the data retrieval process
- QueryRequestParameter: This class will help keep all parameters together as processes move between data managers

**Available Returns**
- Timeseries:
    - EOD or Intraday
    - EOD Can be set to any times, intraday wil only be returned at 1h intervals

- AT Time:
    - At time will use only quotes.
    - Pricing for at time:
        > IF at time is today, it will return intraday pricing
        > IF at time is a different day, it will return EOD Pricing


**Procedures for Processing Timeseries**

***EOD:***
- Step 1: Recieve Timeseries Request
- Step 2: Check if data is currently in database
    - Step 2.1: IF LOGIC
        - If Empty: Query ThetaData and process only to the necessary level. (eg: if vol query, only process to vol, no greeks or attribution). Then produce a thread to process the whole data and save
        - If incomplete: Query ThetaData for min(missing_dates), max(missing_date). Filter for missing dates, and only process that. Save rest to database
        - If Complete: Return to user

***INTRADAY***
- Similar steps to EOD.
- Restrictions:
    - All intraday queries from ThetaData will be 5m, all saves to database will be 5mins as well
        - This is to provide enough granularity, then higher timeframes will utilize resampling
    - Rais error for any timeseries requests <1h.


***DATA CHECKS***
- No Duplicates in final return data
- Index is always datetime
- Dates within a start, end range must be complete
- All columns available, and returned capitalized

**AVAILABLE PARAMETERS:**
- Spots: Midpoint, Weighted Midpoint, Close
- Models: Black Scholes (bsm), Binomial Tree (bt), Monte Carlo Simulation (mcs) (extend this to LSM)
- Produce Vol & Greeks for: Open, Close, Bid, Ask, Mid, Weighted Mid
- Vol Models: BSM & BT

## Future Considerations:
- Clean Up Calc Risks
- Move Binomial, BlackScholes & MonteCarlo to Model Library
- Write the documentation for BulkOptionManager


## save Manager
SaveManager Queue
----------------------
|  Task 1            |
|  Task 2            |  ← enqueue() adds here
|  Task 3            |
----------------------
       ↓
╭────────────╮
│ Thread #1  │ → _worker() gets task → runs save_to_database → loops
╰────────────╯

╭────────────╮
│ Thread #2  │ → same thing
╰────────────╯

╭────────────╮
│ Thread #3  │ → same thing
╰────────────╯

╭────────────╮
│ Thread #4  │ → same thing
╰────────────╯

1. The request is added to cls._queue
2. One of the available worker threads (if idle) picks it up via .get()
3. Runs save_to_database(request)
4. When done, calls .task_done() and loops again

If all 4 threads are busy, the task waits in the queue until a worker becomes free.
