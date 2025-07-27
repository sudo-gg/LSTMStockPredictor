Its in the name really.
For the Data Hackathon by MLH.
Ironically we dont use LSTM and opted for a simpler model,
namely RNN.



Hello world, Im Malachi and this is my demo for the data hackathon.
So we have a login page, it uses sessions to store the username, currently you can just 
put in admin as the username and any password and it will log you in but we will work on
getting mongoDB to store login information and potentially make a sign up page.

So our app is a stock price prediction algorithm, we use a RNN AI model to predict the price of an entered stock and you can enter the name of the company and we use the gemini api to convert the input into a ticker symbol or you can enter the ticker symbol directly.

now for instance let us enter microsoft misspelt.
as you can see it has correctly entered the correct ticker symbol and the model is working.
there are also additional toggles to add external data to our prediction model.

now as you can see it gives us the predicted price and current price at close and it graphs the price of the stock for the week leading up to the current date.

there is also a button to show all companies ticker symbols on the yahoo finance website for when gemini fails and you need to enter the ticker directly, and you dont know the ticker.

and lastly there is a logout button which sends you to the login page and removes session information to enforce you to log back in.