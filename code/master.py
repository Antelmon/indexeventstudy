#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import eventstudy as es 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
#%%
#wdir="./ftse_del/ftse_deletions"
#wdir="./ftse_add/ftse_additions"
wdir="./cac_add/cac_additions"
#wdir="./cac_del/cac_deletions"

#wdir2="./ftse_del"
wdir2="./ftse_add"


#model="market_model"
#model="Fama3"
#model="constant_mean"
#%%
# file_path = wdir+'_returns.csv'

# # Load the CSV file
# df = pd.read_csv(file_path)

# # Function to convert string formatted scientific notation to number
# def convert_scientific_notation(cell):
#     if isinstance(cell, str) and cell.startswith("'"):
#         try:
#             return float(cell[1:])
#         except ValueError:
#             return cell
#     return cell

# # Identify the date column (update with the actual name if different)
# date_column = 'date'

# # Apply the conversion function to all cells in the dataframe except the date column
# for col in df.columns:
#     if col != date_column:
#         df[col] = df[col].apply(convert_scientific_notation)

# # Save the modified dataframe back to the same CSV file
# df.to_csv(file_path, index=False)



#%% merge to market
# df=pd.read_csv(wdir+"_prices.csv")
# df_market=pd.read_csv("MSCI_FRANCE.csv")
# merged_df = pd.merge(df, df_market, on='date', how='inner')
# merged_df.to_csv(wdir+"_prices.csv")
#%% clean

# df=pd.read_csv(wdir+"_prices.csv")

# def clean_dataframe(df):

#     cleaned_df = df.dropna(axis=1, thresh=3)
#     cleaned_df = cleaned_df.dropna(axis=0, thresh=5)
    
#     return cleaned_df
# clean_dataframe(df).to_csv(wdir+"_prices.csv",index=False)
#%%
# def calculate_percentage_returns(df):

#     if 'date' not in df.columns:
#         raise ValueError("DataFrame must contain a 'date' column")
    
#     df = df.set_index('date')
    
#     returns_df = df.pct_change()
    
#     returns_df = returns_df.reset_index()
    
#     returns_df = returns_df.where(pd.notnull(returns_df), None)
    
#     return returns_df

# df=pd.read_csv(wdir+"_prices.csv")
# calculate_percentage_returns(df).to_csv(wdir+"_returns.csv")

# %% FILTERING
df_add=pd.read_csv(wdir+"_volumes.csv")
df_events=pd.read_csv(wdir+"_events.csv")
df_events['event_date']=pd.to_datetime(df_events['event_date'])
df_add['date']=pd.to_datetime(df_add['date'])
print(len(df_events))

df_add=df_add.set_index('date')

def filter_events(prices_df, events_df):
    filtered_events = []

    for _, event in events_df.iterrows():
        ticker = event['security_ticker']
        event_date = event['event_date']
        
        if ticker in prices_df.columns:
            start_date = event_date - pd.Timedelta(days=500)
            end_date = event_date + pd.Timedelta(days=100)
            date_range = pd.date_range(start=start_date, end=end_date)
            
            has_nan = False
            for date in date_range:
                if date in prices_df.index and pd.isna(prices_df.at[date, ticker]):
                    has_nan = True
                    break
            
            if not has_nan:
                filtered_events.append(event)
    
    return pd.DataFrame(filtered_events)


filtered_events_df = filter_events(df_add, df_events)
print(len(filtered_events_df))
filtered_events_df["event_date"]= filtered_events_df["event_date"].dt.strftime('%m/%d/%y')

filtered_events_df.to_csv(wdir+"_events_filtered_for_volumes.csv",index=False)
#%%
def split_date_column_df_and_save(df, date_column="event_date", date_format="%m/%d/%Y"):
    """
    Splits a DataFrame with dates in a specified column into three time-equal portions and saves them in separate directories.

    Parameters:
    df (pd.DataFrame): The DataFrame to split. Must have a column with dates as strings.
    date_column (str): The name of the column containing the dates.
    date_format (str): The format of the date strings.

    Returns:
    tuple: Paths to the saved CSV files.
    """
    global wdir2

    # Convert the date column from string to datetime
    df[date_column] = pd.to_datetime(df[date_column], format=date_format)

    # Ensure the DataFrame is sorted by the date column
    df = df.sort_values(by=date_column)

    # Split into three time-equal portions
    total_periods = len(df)
    split_point1 = int(total_periods / 3)
    split_point2 = int(2 * total_periods / 3)

    # Create the three portions
    df1 = df.iloc[:split_point1]
    df2 = df.iloc[split_point1:split_point2]
    df3 = df.iloc[split_point2:]

    # Print start and end dates of each sub-period
    print("First portion: Start date:", df1[date_column].min(), "End date:", df1[date_column].max())
    print("Second portion: Start date:", df2[date_column].min(), "End date:", df2[date_column].max())
    print("Third portion: Start date:", df3[date_column].min(), "End date:", df3[date_column].max())

    # Convert the date column back to string
    df1[date_column] = df1[date_column].dt.strftime(date_format)
    df2[date_column] = df2[date_column].dt.strftime(date_format)
    df3[date_column] = df3[date_column].dt.strftime(date_format)


    # Define file paths
    early_file = f"{wdir2}/early_sample/early_sample.csv"
    middle_file = f"{wdir2}/middle_sample/middle_sample.csv"
    late_file = f"{wdir2}/late_sample/late_sample.csv"

    # Save the DataFrames as CSV files
    df1.to_csv(early_file, index=False)
    df2.to_csv(middle_file, index=False)
    df3.to_csv(late_file, index=False)


split_date_column_df_and_save(pd.read_csv(wdir+'_events_filtered.csv'))
#%%
es.Single.import_returns(wdir+'_returns.csv', date_format='%m/%d/%Y')
es.Single.import_FamaFrench('./supporting/famafrench-test.csv')
#%%
# test=es.Single.constant_mean(security_ticker="SMT LN Equity",  
# #                            market_ticker="FTSE ALL SHARES",
#                                   event_date=np.datetime64('2017-03-20'),
#                                      event_window = (-40,40), 
#     estimation_size = 100,
#     buffer_size = 40)
# #SMT LN Equity	FTSE ALL SHARES	3/20/2017
# #
# test.plot()
# plt.show()
# print(test.results())
#%%

# %%

model_dict={"market_model" : es.Single.market_model,
            "Fama3"  : es.Single.FamaFrench_3factor,
            "constant_mean" : es.Single.constant_mean
 
}
# workingsubsample= "early_sample"
workingsubsample = "middle_sample"
#workingsubsample = "late_sample"
study = es.Multiple.from_csv(wdir2+'/'+workingsubsample+'/'+workingsubsample+'.csv' ,
                                      model_dict[model], 
                                      date_format = '%m/%d/%Y',
                                      event_window = (-20,20), 
                                      keep_model=True,
    estimation_size = 250,
    buffer_size = 30)
study.error_report()
# %%
study.plot()
plt.savefig("../results/"+wdir2+'/'+workingsubsample+'/'+workingsubsample+'_'+model+".png")
plt.show()
print(study.results())
study.results().to_csv("../results/"+wdir2+'/'+workingsubsample+'/'+workingsubsample+'_'+model)
# %%
l=[]
for i in range(len(study.sample)):
    l.append(np.mean(study.sample[i].model.fittedvalues))
print(np.max(l))
# %%
