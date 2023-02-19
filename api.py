from BOT import Bot
import json
from flask import Flask, request, jsonify

bot = Bot()#Bot is a custom library i created to make my work easy
bot.train()
bot.load_model()





app = Flask(__name__)

@app.route('/<string:msg>/')
def bot_api_resp(msg):
    user_inp = msg
    bot_response =  bot.response(user_inp)
    return jsonify({'user_inp': user_inp,
                    'bot_response': bot_response})
app.run()
    



