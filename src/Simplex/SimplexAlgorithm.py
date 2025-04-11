import numpy as np
from sklearn.neighbors import NearestNeighbors

def simplex_predict(time_series, embedding_dim=4, k=4):
    """
    Predicts the next value in a time series using the Simplex projection method.

    This function uses delay-coordinate embedding to reconstruct state space 
    dynamics from the input time series, finds the `k` nearest neighbours of 
    the most recent embedded point, and returns a weighted average of their 
    subsequent values as the prediction.

    Parameters:
    ----------
    time_series : array-like, shape (n_samples,)
        The input univariate time series.

    embedding_dim : int, default=4
        The number of lagged time steps to use in the embedding (delay dimension).

    k : int, default=4
        The number of nearest neighbours to consider for the prediction.

    Returns:
    -------
    prediction : float
        The predicted next value in the time series.
    """

    # Creates the embedding matrix X using a sliding window
    X = np.array([
        time_series[i:i + embedding_dim]
        for i in range(len(time_series) - embedding_dim)
    ])

    # Target values are the next step after each embedded vector
    y = np.array(time_series[embedding_dim:])

    # Most recent embedded point (i.e. the vector we want to forecast from)
    current_point = time_series[-embedding_dim:]

    # Find k-nearest neighbours to the current point
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors([current_point])

    # Compute weights inversely proportional to distance (with small epsilon to avoid div by 0) and normalise weights to sum to 1
    weights = 1 / (distances.flatten() + 1e-5)
    weights /= weights.sum()

    # Retrieve the corresponding target values for the neighbours
    neighbour_targets = y[indices.flatten()]

    # Weighted average of neighbour targets gives the prediction
    prediction = np.dot(weights, neighbour_targets)

    return prediction


def rolling_simplex_forecast(time_series, embedding_dim=4, k=4, steps=50):
    """
    Performs a rolling forecast using the Simplex projection method.

    This function predicts `steps` future values of a univariate time series,
    one step at a time, by progressively expanding the portion of the time
    series used for prediction. For each step, it uses the Simplex algorithm
    to forecast the next point based on the most recent `embedding_dim` values.

    Parameters:
    ----------
    time_series : array-like, shape (n_samples,)
        The input time series to forecast from.
        
    embedding_dim : int, default=4
        The number of lagged values (dimensions) used to construct the embedding
        vector for prediction.
        
    k : int, default=4
        The number of nearest neighbours to use in the Simplex projection.
        
    steps : int, default=50
        The number of future time steps to forecast.

    Returns:
    -------
    predictions : list of float
        A list containing the forecasted values for the next `steps` time points.

    Notes:
    -----
    This is a rolling one-step-ahead forecast. Each prediction is based only on 
    known past values, mimicking real-world sequential forecasting.
    """
    predictions = []
    
    for i in range(steps):
        input_series = time_series[:-(steps - i)] 
        pred = simplex_predict(input_series, embedding_dim, k)
        predictions.append(pred)
        
    return predictions