# Recommendation System using Machine Learning Algorithms and Python
This project aims to develop a recommendation system using machine learning algorithms and Python. The project uses a sample dataset provided in the data.csv file, which includes information about various items such as their names, descriptions, features, and ratings.

## Installation
To run the code, you will need to install Python 3 and the following libraries:

pandas
scikit-learn
nltk
To install these libraries, you can use pip, a package installer for Python:

```
pip install pandas scikit-learn nltk
```
You will also need to download the nltk data by running the following code in Python:


```
import nltk
nltk.download('stopwords')
```
## Usage
To run the recommendation system, you can execute the recommendation_system.py script:

```
python recommendation_system.py
```
This script reads the data.csv file, preprocesses the data, and creates a similarity matrix using cosine similarity. It then defines a function to recommend items based on the similarity scores and tests the recommendation system by calling the function with a test item id.

You can modify the test item id and the number of recommended items to be returned in the recommend_items() function.

## Contributing
This project is open to contributions. If you have any suggestions or improvements, please feel free to create a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
