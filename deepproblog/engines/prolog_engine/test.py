# Query the model
query = dataset.to_query(0)
result = model.solve([query])[0]
print(result)