# Movie Recommender System (Collaborative Filtering)

## Overview

This project focuses on building a Collaborative Filtering-based movie recommender system. The goal is to suggest movies to users based on their ratings as well as the ratings of other users. The implementation involves the use of a collaborative filtering algorithm, which considers user preferences and movie features to make personalized recommendations.

## 1 - Notation

The following notations are used for referencing and mathematics:

- $r(i,j)$: Scalar; equals 1 if user j rated movie i, 0 otherwise.
- $y(i,j)$: Scalar; rating given by user j on movie i (if $r(i,j) = 1$ is defined).
- $\mathbf{w}^{(j)}$: Vector; parameters for user j.
- $b^{(j)}$: Scalar; parameter for user j.
- $\mathbf{x}^{(i)}$: Vector; feature ratings for movie i.
- $n_u$: Number of users.
- $n_m$: Number of movies.
- $n$: Number of features.
- $\mathbf{X}$: Matrix of vectors $\mathbf{x}^{(i)}$.
- $\mathbf{W}$: Matrix of vectors $\mathbf{w}^{(j)}$.
- $\mathbf{b}$: Vector of bias parameters $b^{(j)}$.
- $\mathbf{R}$: Matrix of elements $r(i,j)$.

## 2 - Recommender Systems

Collaborative filtering involves generating two vectors for each user and movie: a 'parameter vector' for user preferences and a feature vector for movie descriptions. The recommender system predicts a user's rating for a movie by taking the dot product of these vectors and adding a bias term.

## 3 - Movie Ratings Dataset

The dataset used is derived from the MovieLens "ml-latest-small" dataset [https://grouplens.org/datasets/movielens/latest/], focusing on movies from the years since 2000. It includes ratings on a scale of 0.5 to 5 in 0.5 steps. The reduced dataset has 443 users and 4778 movies.

## 4 - Collaborative Filtering Learning Algorithm

### 4.1 Collaborative Filtering Cost Function

The collaborative filtering cost function is defined as:

The collaborative filtering cost function is given by:

\[ J(\mathbf{x}, \mathbf{w}, b) = \frac{1}{2} \sum_{(i,j):r(i,j)=1} \left( (\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)})^2 \right) + \frac{\lambda}{2} \sum_{j=0}^{n_u-1} \sum_{k=0}^{n-1} (\mathbf{w}^{(j)}_k)^2 + \frac{\lambda}{2} \sum_{i=0}^{n_m-1} \sum_{k=0}^{n-1} (\mathbf{x}_k^{(i)})^2 \]

This cost function represents the error in predicting user ratings for movies. The first term measures the squared difference between predicted and actual ratings for movies that have been rated. The second and third terms are regularization terms to prevent overfitting. The hyperparameter \(\lambda\) controls the strength of regularization.

## 5 - Learning Movie Recommendations

The collaborative filtering model is trained using TensorFlow's custom training loop. The parameters $\mathbf{X}$, $\mathbf{W}$, and $\mathbf{b}$ are learned to minimize the cost function.

## 6 - Recommendations

The trained model is used to make movie recommendations based on user ratings. The provided user ratings are used to predict ratings for all movies, and the top suggestions are displayed. These suggestions are personalized to the user's preferences.

## Conclusion

The collaborative filtering recommender system successfully provides relevant movie suggestions based on user ratings. This system can be integrated into applications to enhance user experience and engagement.
