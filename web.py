from flask import Flask, request
import captcha_parser
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def getCaptchaCode():
  return captcha_parser.getCaptchaCode(request.files['file'].stream)

if __name__ == '__main__':
  app.run()