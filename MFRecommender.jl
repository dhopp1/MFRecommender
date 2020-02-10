module MFRecommender
    using Recommendation

    mutable struct GenRecommender
        raw_user_ids::Array
        raw_media_ids::Array
        ratings::Array{Float64}
        user_dict::Dict{Any, Int64}
        media_dict::Dict{Any, Int64}
        function GenRecommender(raw_user_ids, raw_media_ids, ratings)
            users = raw_user_ids |> unique
            user_dict = zip(users, 1:length(users)) |> Dict
            media = raw_media_ids |> unique
            media_dict = zip(media, 1:length(media)) |> Dict
            new(raw_user_ids, raw_media_ids, ratings, user_dict, media_dict)
        end
    end
    GenRecommender(x, y) = GenRecommender(x, y, fill(10.0, (length(x), 1))) # default value of 10.0 for ratings

    function gen_events(r::GenRecommender)
        events = Event[]
        for (user, media, rating) in zip(r.raw_user_ids, r.raw_media_ids, r.ratings)
            push!(events, Event(r.user_dict[user], r.media_dict[media], rating))
        end
        return events
    end

    gen_data_accessor(e::Array, r::GenRecommender) = DataAccessor(e, length(r.user_dict), length(r.media_dict))

    function train_recommender(r::GenRecommender; reg::Float64=1e-3, learning_rate::Float64=1e-3, eps::Float64=1e-3, max_iter::Int=100)
        mf = gen_events(r) |> x -> gen_data_accessor(x, r) |> MF
        build!(mf, reg=reg, learning_rate=learning_rate, eps=eps, max_iter=max_iter)
        return mf
    end

    struct rec
        rank::Int
        media_id
        score::Float64
    end

    function inference(r::GenRecommender, m::MF, user_id, n_recs::Int)
        recs = recommend(m, haskey(r.user_dict, user_id) ? r.user_dict[user_id] : user_id, n_recs, collect(1:length(r.user_dict)))
        recs = [([k for (k, v) in filter(p -> last(p) == key, r.media_dict)][1], value) for (key, value) in recs]
        final_recs = []
        for i in enumerate(recs)
            push!(final_recs, rec(i[1], i[2][1], i[2][2]))
        end
        return final_recs
        return recs
    end

    function initialize(user_ids::Array, media_ids::Array, ratings::Array=nothing; reg::Float64=1e-3, learning_rate::Float64=1e-3, eps::Float64=1e-3, max_iter::Int=100)
        _recommender = GenRecommender(user_ids, media_ids, ratings)
        trained_recommender = train_recommender(_recommender, reg=reg, learning_rate=learning_rate, eps=eps, max_iter=max_iter)
        return (_recommender, trained_recommender)
    end

    function initialize(user_ids::Array, media_ids::Array; reg::Float64=1e-3, learning_rate::Float64=1e-3, eps::Float64=1e-3, max_iter::Int=100)
        _recommender = GenRecommender(user_ids, media_ids)
        trained_recommender = train_recommender(_recommender, reg=reg, learning_rate=learning_rate, eps=eps, max_iter=max_iter)
        return (_recommender, trained_recommender)
    end

end
