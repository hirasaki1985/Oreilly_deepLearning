# -*- coding: utf-8 -*-

import sys, socket
import json
import cgi
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from controllers.controller import Controller
from neuralNetwork.Backpropagation import BackPropagation
from controllers.simpleConvController import SimpleConvController

# setting
host = ''
port = 8000

class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print("simpleserver do_POST exec()")

        # init
        #self.controller = Controller(network = BackPropagation())
        self.controller = SimpleConvController()
        #self.allowOrigin = 'http://deeplearning.local.com'
        #self.allowOrigin = "http://" + self.headers.get('Host').rsplit(":")[0]
        self.allowOrigin = "http://127.0.0.1:8888"

        # access check
        if self.path.endswith('favicon.ico'):
          return;

        # request
        form = self.getRequestData()
        print(type(form))

        # logic
        #logicResult = ""
        logicResult = self.controller.webLogic(form)

        # make result
        result = self.makeResponseData(logicResult)

        # send
        self.sendResponse(result)
        return

    def getRequestData(self):
        # POST されたフォームデータを解析する
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':'png',
                    })
        print(form)
        #image = {"test":"requestData"}
        return form

    def makeResponseData(self, result):
        print("### simpleserver makeResponseData exec")
        #result = {"test":"responseData"}

        print(result)
        print(type(result))

        return result

    def sendResponse(self, result):
        print("### simpleserver sendResponse exec")
        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.send_header('Access-Control-Allow-Origin', self.allowOrigin)
        self.end_headers()

        #self.wfile.flush()
        self.wfile.write(str(result).encode('UTF-8'))
        self.wfile.close()
        return

try:
    server = HTTPServer((host, port), MyHandler)
    server.serve_forever()

except KeyboardInterrupt:
    print ('^C received, shutting down the web server')
    server.socket.close()
