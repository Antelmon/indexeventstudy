#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import eventstudy as es 
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
#%%
#wdir="./ftse_del/ftse_deletions"
#wdir="./ftse_add/ftse_additions"
wdir="./cac_add/cac_additions"
#wdir="./cac_del/cac_deletions"

#wdir2="./ftse_del"
#wdir2="./ftse_add"
#%%
import pandas as pd

def empty_repetitions(df, threshold=10):
    """
    Empties any cell in the DataFrame that is part of a repetition of more than 'threshold' cells.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (int): The threshold for the number of repetitions.

    Returns:
    pd.DataFrame: The modified DataFrame with cells emptied.
    """
    for column in df.columns:
        count = 1
        for i in range(1, len(df)):
            if df.at[i, column] == df.at[i-1, column]:
                count += 1
            else:
                if count > threshold:
                    df.loc[i-count:i-1, column] = None
                count = 1
        if count > threshold:  # Handle case where the repetition ends at the last row
            df.loc[len(df)-count:len(df)-1, column] = None
    return df

# Load the CSV file into a DataFrame
input_file = wdir+"_volumes.csv"
df = pd.read_csv(input_file)

# Process the DataFrame to empty repeated cells
df = empty_repetitions(df)

# Save the modified DataFrame back to a CSV file
output_file =wdir+"_volumes.csv"
df.to_csv(output_file, index=False)



# %%
# Load the event DataFrame from a CSV file
event_file_path = wdir+"_events_filtered_for_volumes.csv"
event_df = pd.read_csv(event_file_path)
# Load the volumes traded DataFrame from a CSV file
volume_file_path = wdir+'_volumes.csv'
volume_df = pd.read_csv(volume_file_path, index_col=0)

def plot_distribution_of_estimation_window_volumes(event_df, volume_df):
    """
    Plots the distribution graph of all the normalized volumes in the estimation window.

    Parameters:
    event_df (pd.DataFrame): DataFrame containing columns 'ticker' and 'event date'.
    volume_df (pd.DataFrame): DataFrame with date as index and tickers as columns containing traded volumes.
    """
    # Convert the 'event date' column to datetime
    event_df['event_date'] = pd.to_datetime(event_df['event_date'],format='%m/%d/%y')
    # Ensure the index of volume_df is in datetime format
    volume_df.index = pd.to_datetime(volume_df.index)

    estimation_volumes = []
    event_volumes =[]
    for _, row in event_df.iterrows():
        ticker = row['security_ticker']
        event_date = row['event_date']

        # Define the estimation window (150 days before the 30-day buffer period)
        estimation_start = event_date - pd.Timedelta(days=210)
        estimation_end = event_date - pd.Timedelta(days=51)

        # Get the estimation period data
        estimation_period = volume_df.loc[estimation_start:estimation_end, ticker]
        event_period=volume_df.loc[event_date-pd.Timedelta(days=20):event_date+pd.Timedelta(days=1),ticker]
        
        event_period_off=pd.Series(data=(event_period.index - event_date),index=event_period.index)
        event_period =pd.concat([event_period,event_period_off], axis=1)
        event_period.rename(columns={ticker :"volume"}, inplace=True)
        # Calculate mean and standard deviation
        mean_volume = estimation_period.mean()
        std_volume = estimation_period.std()

        # Normalize the volumes in the estimation period
        normalized_estimation_volumes = (estimation_period - mean_volume) / std_volume

        event_period.volume=(event_period.volume-mean_volume)/ std_volume
        event_period = event_period[event_period['volume'] == event_period['volume'].max()]

        estimation_volumes.append(normalized_estimation_volumes)
        event_volumes.append(event_period)
    


    pd.concat(estimation_volumes).to_csv("../results/"+wdir+'_estimation_volumes.csv')
    pd.concat(event_volumes).to_csv("../results/"+wdir+'_event_volumes.csv')
        
    plt.figure(figsize=(10, 6))
    sns.histplot(pd.concat(estimation_volumes), bins=30, kde=True)
    plt.title('Distribution of Normalized Trading Volumes')
    plt.xlabel('Normalized Volume')
    plt.ylabel('Frequency')
    plt.savefig('../results/'+wdir+'_volumes_graph.png')
    plt.show()

    # Plot the distribution
        

    
 

plot_distribution_of_estimation_window_volumes(event_df, volume_df)




# %%
