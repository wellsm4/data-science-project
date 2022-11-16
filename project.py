import plotly.express as px
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning) # Suppress extraneous warning
import pandas
import numpy
import scipy.stats as stats

# Read data
df = pandas.read_csv("./data/dataset.csv", dtype={"price": str, "rating": str, "n_reviews": str})

# Clean data
df = df[df.price != "Not available"]               #
df = df[df.rating != "Not available"]              # Expunge null values
df = df[df.n_reviews != "No customer reviews yet"] #
df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype("float") # Remove dollar signs and commas from prices
df["rating"] = df["rating"].astype("float")
df["n_reviews"] = df["n_reviews"].str.replace(",", "").astype("int") # Remove commas from review counts

# Initial visualizations
vis_1 = px.scatter(data_frame = df, title = "Price versus rating", x = "price", y = "rating") # Scatterplot of price by rating
vis_2 = px.scatter(data_frame = df, title = "Price versus number of reviews", x = "price", y = "n_reviews") # Scatterplot of price by number of reviews
vis_3 = px.scatter(data_frame = df, title = "Number of reviews versus rating", x = "n_reviews", y = "rating") # Scatterplot of number of reviews by rating
vis_1.show()
vis_2.show()
vis_3.show()

# Model 1: polyfit
x_data_1 = df["price"].to_numpy() # Convert price column to numpy format
x_data_1 = numpy.sort(x_data_1) # Sort x data ascending
y_data_1 = df["rating"].to_numpy() # Convert rating column to numpy format
y_data_1 = numpy.sort(y_data_1) # Sort y data ascending
x_data_log_1 = numpy.log(x_data_1) # Take the natural log of all the x data
curve_1 = numpy.polyfit(x_data_log_1, y_data_1, 1) # Find curve
print("Logarithmic equation of best fit: y = " + curve_1[0].astype("str") + "ln(x) + " + curve_1[1].astype("str"))
ml_predictions = curve_1[0] * x_data_log_1 + curve_1[1] # Make predictions based on curve

# Visualization 4 (of predictions)
vis_4 = px.line(title="Regression curve for price vs. rating", x = x_data_1, y = ml_predictions, labels=dict(x = "price", y = "rating"))
vis_4.show()

# Model 2: Goodness of fit
chi_sq_test_stat, p_value = stats.chisquare(y_data_1, ml_predictions) # Run chi-square model
chi_sq_crit_val = stats.chi2.ppf(1-0.05, df=(len(y_data_1) - 1)) # Calculate critical value with 95% confidence interval and n - 1 degrees of freedom
if chi_sq_crit_val >= chi_sq_test_stat: # Compare test statistic to critical value
	print("Null hypothesis accepted; data reasonably fits projection")
else:
	print("Null hypothesis rejected; data may not reasonably fit projection")

# Groupbys
n_by_rating = df.groupby("rating")["rating"].count() # Group ratings by number of each
vis_5 = px.scatter(data_frame = n_by_rating, title="Rating by number of ratings")
vis_5.show()

n_by_price = df.groupby("price")["price"].count() # Group price by number of each
vis_6 = px.scatter(data_frame = n_by_price, title="Price by number of prices")
vis_6.show()

# Join
oldest_100 = df.head(100)
newest_100 = df.tail(100)
old_n_new = pandas.merge(oldest_100, newest_100, on="name", how="inner") # Inner join first and last 100 entries on name
print("Products common to oldest and newest 100: " + old_n_new["name"].count().astype("str"))