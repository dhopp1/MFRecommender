include("MFRecommender.jl")

using DataFrames, XLSX

data = XLSX.readxlsx("data.xlsx") |>
    x -> x[XLSX.sheetnames(x)[1]][x[XLSX.sheetnames(x)[1]].dimension] |>
    DataFrame |>
    x -> rename!(x, Symbol.(convert(Array, x[1,:]))) |>
    x -> x[2:end, :]

model = MFRecommender.initialize(data.user_ids, data.media_ids)

MFRecommender.inference(model..., "17491", 10)
