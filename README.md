# Term Deposit Marketing
## Background:

ACME is a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.

We are interested in developing a robust machine learning system that leverages information coming from call center data.

Ultimately, at ACME we are looking to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

## Data Description:

The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

## Attributes:

age : age of customer (numeric)

job : type of job (categorical)

marital : marital status (categorical)

education (categorical)

default: has credit in default? (binary)

balance: average yearly balance, in euros (numeric)

housing: has a housing loan? (binary)

loan: has personal loan? (binary)

contact: contact communication type (categorical)

day: last contact day of the month (numeric)

month: last contact month of year (categorical)

duration: last contact duration, in seconds (numeric)

campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

## Output (desired target):

y - has the client subscribed to a term deposit? (binary)

## Goal(s):

Predict if the customer will subscribe (yes/no) to a term deposit (variable y)

## Success Metric(s):

Hit %81 or above accuracy by evaluating with 5-fold cross validation and reporting the average performance score.

## Bonus(es):

We are also interested in finding customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.

What makes the customers buy? Tell us which feature we should be focusing more on.

## Project requirements
python==3.8
numpy==1.19.2
matplotlib==3.3.2
scikit-learn==0.23.2
pandas==1.1.3

## Project Outputs
In the project, we looked at whether the customer would buy the product with a random data entry. According to the data we entered, our model gave the output ('no') that the customer would not buy the product.We also calculated that our model works with 92.96% accuracy.

![output1](https://user-images.githubusercontent.com/52162324/99885027-2cbdcf80-2c43-11eb-8b68-8ba823dc86d1.PNG)

#### BONUS PARTS
In bonus episodes,
-What makes the customers buy? Tell us which feature we should be focusing more on?
-Finding customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
we solved the problems above.

We displayed the 10 most important features in the data set.

![output2](https://user-images.githubusercontent.com/52162324/99885251-a30f0180-2c44-11eb-8a16-639d47886249.PNG)

We displayed the features of the customers who had the potential to buy the product.

SUMMARY FOR NUMERİCAL VALUES
![output3_1_1](https://user-images.githubusercontent.com/52162324/99885354-3c3e1800-2c45-11eb-84d0-71dfcfbe985a.PNG)
SUMMARY FOR OBJECT VALUES
![output3_2_1](https://user-images.githubusercontent.com/52162324/99885362-495b0700-2c45-11eb-852c-0464eb4b58e3.PNG)



