# MFRecommender
Julia MF recommender using the Recommendation package

## Background
Uses [Recommendation.jl](https://github.com/takuti/Recommendation.jl) to create a [matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) recommender. The motivation was to create a very simple syntax to generate a recommender with an array of user ids and a corresponding array of media ids (optionally ratings as well).

## Usage
```
include("MFRecommender.jl")
model = MFRecommender.initialize(user_ids, media_ids, ratings) # if ratings exist
model = MFRecommender.initialize(user_ids, media_ids) # rating of 10.0 applied to all
MFRecommender.inference(model..., user_id, n_recs)
```