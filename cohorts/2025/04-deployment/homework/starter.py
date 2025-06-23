import pickle
import pandas as pd
import numpy as np 
import sys


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def apply_model(year, month):
    with open('homework/model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'duration_pred': y_pred})

    output_file = f'./output_yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print("mean predicted duration: ", np.mean(y_pred))


def run():
    year = int(sys.argv[1]) # 2023
    month = int(sys.argv[2]) # 3

    apply_model(
        year=year,
        month=month
    )


if __name__ == '__main__':
    run()
