## Pointers
- Can create an overall Strategy Class
- For each test, new classes can inherit the overall strategy and override where necessary


## Pre-req
Designing the idea Strategy
Caveat: This stage is outlining what we want the strategy to look like. It hasn't gotten into design. It focuses on understanding the important parts of the strategy and what it should do.
1. Set a smart goal: What is this strategy intended to achieve, it's properties, it's risk type, trade duration, etc.
2. How will you enter, how will you exit
3. What time frame & bar size will the trade occur?


## STRATEGY BASE STEPS: Initial Verification tests
Caveat: This stage involves initial design. We take the strategy and see if there is something there to begin with. It is a simple simulation.
Step 1: Create logic using the rules of StrategyBase
Step 2: Test a plain simulation using StrategyBase

## PT BACKTESTER STEPS: Event Driven Equity Test
Caveat: Initial test to see if there's a slimmer of robustness when doing a small in/out sampling 
1. Split Test: Split into train & test years. Use the trained parameters to test the test years. 
    1.1. Criteria: Return, Sharpe, Expectancy, Win Rate

### Entry Testing: We want to know if entry provides any usefulness
Caveat: We're testing our entries and seeing if it is good or can be improved on
a. Fixed-Stop and Target Exit:
    - Set a stop same distance as your take profit.
    - Optimize and use best params.
    - ***GOAL:*** All else equal, you should prevail about 50% ofthe time.

b. Fixed-Bar Exit:
    - Set an exit x days after entry.
    - ***GOAL:*** Do we show profit soon? If not, is entry bad? Could it be improved on? We want to know if entry signals are bad

c. Random Exit:
    - Check if profits are generated on a random exit.
    - ***GOAL:*** If profit is generated, there are chances this is good

d. Delayed Entry:
    - Lag days from 1-5 days
    - Randomize entry for individual days

### Exit Testing:
Caveat: We're testing our exits and seeing if it is good or can be improved on
a. Similar Approach:
    - Create other forms of entry for same risk factor, and see if entry signal improves or not

b. Random Entries:
    - Enter at random and see what performance looks like (Need to think more about this)


### Filter Testing
Caveat: We're testing our filters and seeing if it is good or can be improved on
a. Simple Approach: Take the base from previous development and see if adding a filter improves the strategy

### Core System aka Parameter Testing:
Caveat: Bringing the system together and seeing it's performance
a. Test an array of parameters and see the heatmap (This should be strategy level parameters)
b. Run another in/out test to ascertain robustness

### Walk Forward Analysis:
Caveat: Concept is straight forward, split into train and test. Train, test and move forward. The following is what we will be testing. We are looking to learn how robust our strategy really is. And how optimization affects it over time

a. Target Optimiation: What's the right stuff to be optimizing for?
b. Timeframe Selection: What is appropriate train/test combo?
c. Weight Optimization: What is the appropriate way to distribute cash btwn tickers? If they're more than 1.
d. Anchored vs Unachored

### Monte Carlo Simulation:
Caveat: Deeper learning on the statistics/strengths/weaknesses of the strategy of our strategy.
- Run the monte carlo simulation and produce the key statistics on the strategy.



## Option Testing:
Caveat: For we need to first create a base. Base aggregation, eq, trades, greeks, pnl attr. Then we run an analysis to see if the improvements are statistical
### Delayed Entry & exit:
Caveat: We want to see how sliding entry affects performance.
- Static: Slide t_plus_n from 1 - 5
- Random: Randomize slide. Randomize delayed entry by 1-5

### Commission/Slippage:
- Still basic for now. How does different commission/Slippage affect the strategy

### Delta Sizing & Dynamic Sizing:
- 1. Strategy Leverage: Can this be dynamic?
- 2. Does limit sizing or cash available sizing make a difference?
- 3. Should we have limits enabled?
- 4. Does dynamic sizing make a difference? Does it improve or worsen?

### Contract Type:
- 1. Does DTE affect?
- 2. Is naked better than vertical?
- 3. How does open interest change affect the strategy?
- 4. OTM or ATM?

### WFA:
- 1. Ultimately, WFA will bring together all saved information and produce an equity curve to get a sense on how it performs.
- 2. Based on the above, it can variate the parameters to find the best

### Monte Carlo Simulation:
- Same as stock

## Saving Statistics.
- Information to save (All stats will only be WFA combined):
    - Aggregate stats eq
    - Equity curve
    - Monte Carlo stats eq
    - Trades eq
    - Aggregate stats option
    - Monte Carlo stats option
    - Equity Curve option
    - Trade Options
    - Monthly performance for both Eq & Options

## Notes:
- Strategy will start with signal on date, no entry on miss.
- We will then test lagged entries and decide a limit for when to enter a strategy. This will be max_dates_lag
- Also test how lagging exits deteriorates the strategy
