# Movie Recommendation System

This is a movie recommendation system built using Python and Streamlit. It recommends similar movies based on the user's input movie title.

## Installation

1. Clone or download the repository:

    ```bash
    git clone https://github.com/your-username/movie-recommendation-system.git
    ```

2. Navigate to the project directory:

    ```bash
    cd movie-recommendation-system
    ```

3. Download the dataset from the following link and place the CSV files in the `data` directory:

    [Movie Dataset](https://drive.google.com/drive/folders/1sPRn2LiE4bO1H2K_a3MkvVl6XVUGrqKJ)

    The dataset consists of two CSV files:
    - `credits.csv`: Contains information about the cast and crew of each movie.
    - `movies.csv`: Contains information about each movie, including its title, genres, keywords, overview, etc.

4. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Make sure you have Python installed on your machine.

2. Run the Streamlit app:

    ```bash
    streamlit run movie_recommendation_app.py
    ```

3. Open a web browser and navigate to the URL displayed in the terminal.

4. Enter a movie title in the input field and click the "Get Recommendations" button to see similar movie recommendations.

## Features

- User-friendly interface for entering movie titles and viewing recommendations.
- Utilizes TF-IDF vectorization and k-nearest neighbors algorithm for movie recommendation.
- Supports customization of the number of nearest neighbors to consider.

