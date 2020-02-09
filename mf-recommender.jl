using DataFrames,
Recommendation,
XLSX


mutable struct MFRecommender
    raw_user_ids::Array
    raw_media_ids::Array
    ratings::Array{Float64}
    user_dict::Dict{Any, Int64}
    media_dict::Dict{Any, Int64}
    function MFRecommender(raw_user_ids, raw_media_ids, ratings)
        users = raw_user_ids |> unique
        user_dict = zip(users, 1:length(users)) |> Dict
        media = raw_media_ids |> unique
        media_dict = zip(media, 1:length(media)) |> Dict
        new(raw_user_ids, raw_media_ids, ratings, user_dict, media_dict)
    end
end
MFRecommender(x, y) = MFRecommender(x, y, fill(10.0, (length(x), 1))) # default value of 10.0 for ratings

function gen_events(r::MFRecommender)
    events = Event[]
    for (user, media, rating) in zip(r.raw_user_ids, r.raw_media_ids, r.ratings)
        push!(events, Event(r.user_dict[user], r.media_dict[media], rating))
    end
    return events
end

gen_data_accessor(e::Array, r::MFRecommender) = DataAccessor(e, length(r.user_dict), length(r.media_dict))

function train_recommender(r::MFRecommender, max_iter::Int)
    mf = gen_events(r) |> x -> gen_data_accessor(x, r) |> MF
    build!(mf, max_iter=max_iter)
    return mf
end


# data read, temporary with data
data = XLSX.readxlsx("data_recommender.xlsx") |>
    x -> x[XLSX.sheetnames(x)[1]][x[XLSX.sheetnames(x)[1]].dimension] |>
    DataFrame |>
    x -> rename!(x, Symbol.(convert(Array, x[1,:]))) |>
    x -> x[2:end, :]

# training and inference
_recommender = MFRecommender(data.user_steamid, data.appid)
trained_recommender = train_recommender(_recommender, 50)
recommend(trained_recommender, 56, length(_recommender.user_dict), collect(1:length(_recommender.user_dict)))
